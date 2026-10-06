# The one-collective skew acceptance for the rotating arrival banks.
#
# A side that falls one collective behind must still find the id it is waiting for. The engine
# produces that skew on its own: wherever a round reads its sample only shard A is drained, so the
# mirror shard is routinely a call or two behind, and the in-kernel allreduce's arrival slot is
# overwritten by the peer's next collective. Before the banks that made the wait unsatisfiable and the
# bounded spin gave up after its deadline (HTTP 503, "TP-2 allreduce stalled"); with the banks the id
# survives in the bank it was published under.
#
# The injection holds one side's arrival poll of one eager collective
# (NINFER_TP2_AR_FAULT_HOLD_POLL=<eager serial>:<nanos>), so the other side completes that collective
# and publishes the next one first: deterministically the same skew the production ar-watch dumps show
# (an A/B gap of exactly one id in arrival slot 0). Both arms run the same workload with the same
# injection and differ only in NINFER_TP2_AR_BANKS:
#
#   control (1 bank)   the pre-bank geometry: the run must fail - the warmup with a stall, or the
#                      request with 503 and a watchdog dump
#   banks   (4 banks)  the fix: the server must start and serve every request, no stall reported
#
#   pwsh -File tools/tp_bootstrap/r89_skew_acceptance.ps1
#   pwsh -File tools/tp_bootstrap/r89_skew_acceptance.ps1 -Call 400 -HoldMs 50
#
# The default serial (100) lands inside the warmup's eager collectives, so the control arm fails
# before the engine is up; an arm whose serial was never reached reports INVALID and the arm above it
# shows how far the run got.
#
# The server runs from build-win/apps/ninfer-serve.exe (build it first: tools/win_port/build.ps1) with
# the production TP-2 DFlash2 recipe, and only the process this script started is stopped. Raise -Call
# when the injection lands in the warmup; the arm then reports "FATAL warmup failed" as INVALID.
[CmdletBinding()]
param(
    [string] $Model = 'D:/LLM/rloo351-mixed-mtp-dflash2.ninfer',
    [string] $Binary = '',
    [string] $ChatTemplate = 'D:/LLM/chat_template.jinja',
    [int]    $Port = 8099,
    [int]    $Concurrency = 2,
    [int]    $DraftTokens = 5,
    [int]    $Context = 131072,
    [int]    $HostKvMiB = 20480,
    [int]    $Call = 100,
    [int]    $HoldMs = 30,
    [int]    $Requests = 2,
    [int]    $StartupTimeoutSec = 300,
    [string] $OutDir = 'build-win/r89',
    [string] $FfmpegRoot = 'D:/ffmpeg-dev/expanded/ffmpeg-master-latest-win64-gpl-shared',
    [string] $LibcurlRoot = 'D:/curl-dev/expanded/curl-8.22.0_1-win64-mingw'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
if (-not $Binary) { $Binary = Join-Path $repo 'build-win/apps/ninfer-serve.exe' }
if (-not (Test-Path $Model))  { throw "model artifact not found: $Model" }
if (-not (Test-Path $Binary)) { throw "server binary not found: $Binary (run tools/win_port/build.ps1)" }
if (@(Get-Process -Name ninfer-serve -ErrorAction SilentlyContinue).Count -ne 0) {
    throw 'a ninfer-serve process is already running; stop it first (it owns the TP-2 pair)'
}

# ninfer-serve loads avcodec/swscale and libcurl at run time, and build-win/apps does not carry them,
# so both DLL directories have to be on PATH or the server never reaches the GPU.
$env:PATH = "$FfmpegRoot/bin;$LibcurlRoot/bin;$env:PATH"

$prompt = 'A train leaves Station A at 09:15 travelling 72 km/h, and a second train leaves Station B at 09:40 travelling 96 km/h. The stations are 310 km apart. Solve this step by step, showing every calculation and checking the result.'
$script:server_pid = 0

function Wait-Health([int] $Seconds) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        if (-not (Get-Process -Id $script:server_pid -ErrorAction SilentlyContinue)) { return 'process-gone' }
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5 -UseBasicParsing
            if ($r.StatusCode -eq 200) { return 'ready' }
        } catch { }
        Start-Sleep -Seconds 3
    }
    return 'timeout'
}

function Invoke-Arm {
    param([string] $Name, [int] $Banks, [bool] $ExpectStall)

    $log = Join-Path $OutDir ("skew-" + $Name + ".log")
    Remove-Item $log, "$log.out" -ErrorAction SilentlyContinue
    # Start-Process joins this array with spaces and does not quote elements, so every path here has
    # to be free of spaces (the model, the binary and the chat template all are on this machine).
    $arguments = @(
        $Model, '--devices', '0,1', '--max-context', "$Context", '--port', "$Port",
        '--host', '127.0.0.1', '--kv-dtype', 'int8', '--log-level', 'info',
        '--max-concurrency', "$Concurrency", '--prefill-chunk', '1024',
        '--max-pending-requests', '16', '--host-state-slots', '32',
        '--host-kv-mib', "$HostKvMiB", '--max-private-continuations', '6',
        '--session-retention-floor', '4096', '--spec', 'dflash2',
        '--draft-tokens', "$DraftTokens", '--lm-head-draft',
        '--reasoning-effort', 'xhigh', '--preserve-thinking'
    )
    if ($ChatTemplate -and (Test-Path $ChatTemplate)) { $arguments += @('--chat-template', $ChatTemplate) }

    # The child inherits these; the server reads them when it builds its DevicePair.
    $env:NINFER_TP2_AR_BANKS = "$Banks"
    $env:NINFER_TP2_AR_FAULT_HOLD_POLL = ('{0}:{1}' -f $Call, ([int64]$HoldMs * 1000000))
    $env:NINFER_TP2_AR_WATCHDOG = '1'
    $process = Start-Process -FilePath $Binary -ArgumentList $arguments -NoNewWindow -PassThru -RedirectStandardOutput "$log.out" -RedirectStandardError $log
    $script:server_pid = $process.Id
    Remove-Item Env:NINFER_TP2_AR_BANKS, Env:NINFER_TP2_AR_FAULT_HOLD_POLL, Env:NINFER_TP2_AR_WATCHDOG -ErrorAction SilentlyContinue

    $statuses = @()
    try {
        $state = Wait-Health $StartupTimeoutSec
        if ($state -ne 'ready') {
            $statuses += ("start:" + $state)
        } else {
            $modelId = (Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/models" -TimeoutSec 30).data[0].id
            for ($i = 0; $i -lt $Requests; $i++) {
                if (-not (Get-Process -Id $script:server_pid -ErrorAction SilentlyContinue)) {
                    $statuses += 'process-gone'; break
                }
                $payload = @{
                    model = $modelId
                    messages = @(@{ role = 'user'; content = $prompt })
                    max_tokens = 256; temperature = 0.7; top_k = 20; top_p = 0.8; stream = $false
                } | ConvertTo-Json -Depth 8 -Compress
                try {
                    $request = @{
                        Uri = "http://127.0.0.1:$Port/v1/chat/completions"
                        Method = 'Post'
                        ContentType = 'application/json; charset=utf-8'
                        Body = [System.Text.Encoding]::UTF8.GetBytes($payload)
                        TimeoutSec = 300
                    }
                    $null = Invoke-RestMethod @request
                    $statuses += 'ok'
                } catch {
                    $msg = ($_.ErrorDetails.Message + ' ' + $_.Exception.Message)
                    if ($msg -match 'service_unavailable' -or $msg -match 'stalled') { $statuses += '503' }
                    elseif ($msg -match 'actively refused|Unable to connect') { $statuses += 'conn-refused' }
                    else { $statuses += ('err:' + $msg.Substring(0, [Math]::Min(80, $msg.Length))) }
                }
            }
        }
    } finally {
        if ($script:server_pid) {
            if (Get-Process -Id $script:server_pid -ErrorAction SilentlyContinue) {
                Stop-Process -Id $script:server_pid -Force
                Start-Sleep -Seconds 3
            }
        }
    }

    $text = if (Test-Path $log) { Get-Content $log -Raw } else { '' }
    $dump      = [bool]($text -match '\[ar-watch\]')
    $tripped   = [bool]($text -match 'stalled at rendezvous|IllegalAddress|illegal memory')
    $warmup    = [bool]($text -match 'FATAL warmup failed')
    $armed     = [bool]($text -match ('\[ar-fault\] eager collective ' + $Call + ':'))
    $banksLine = ([regex]::Match($text, 'arrival banks: \d+')).Value

    $verdict = 'as-expected'
    if (-not $armed) {
        $verdict = ('INVALID: eager collective ' + $Call + ' was never issued; check -Call')
    } elseif ($ExpectStall) {
        # Either evidence is the pre-bank behavior: the skew kills the warmup (the engine exits with a
        # stall message) or it lands in a request (503 plus the watchdog dump).
        if ($warmup -and $tripped) { }
        elseif (($statuses -contains '503') -and $dump) { }
        elseif ($warmup) { $verdict = 'UNEXPECTED: the warmup failed without a stall message' }
        elseif (-not ($statuses -contains '503')) { $verdict = 'UNEXPECTED: the control did not fail with 503' }
        else { $verdict = 'UNEXPECTED: the control failed without a watchdog dump' }
    } else {
        if ($warmup) { $verdict = 'UNEXPECTED: the warmup failed with the banks in place' }
        elseif ($statuses -contains '503' -or $statuses -contains 'process-gone') { $verdict = 'UNEXPECTED: the request failed' }
        elseif ($dump) { $verdict = 'UNEXPECTED: a stall was reported with the banks in place' }
        elseif ($statuses -notcontains 'ok') { $verdict = 'UNEXPECTED: no request succeeded' }
    }

    Write-Host ('ARM ' + $Name.PadRight(8) + ' banks=' + $Banks + ' -> ' + ($statuses -join ',') + ' | dump=' + $dump + ' tripped=' + $tripped + ' armed=' + $armed + ' | ' + $verdict)
    Write-Host ('  log ' + $log + '  ' + $banksLine + '  ar-watch lines=' + ([regex]::Matches($text, '\[ar-watch\]').Count))
    return [pscustomobject]@{
        arm = $Name; banks = $Banks; statuses = ($statuses -join ','); dump = $dump
        tripped = $tripped; armed = $armed; warmup = $warmup; verdict = $verdict; log = $log
    }
}

$rows = @()
$rows += Invoke-Arm -Name 'control' -Banks 1 -ExpectStall $true
$rows += Invoke-Arm -Name 'banks'   -Banks 4 -ExpectStall $false
$rows | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $OutDir 'skew-summary.json') -Encoding UTF8

$bad = @($rows | Where-Object { $_.verdict -ne 'as-expected' })
Write-Host ''
Write-Host ('skew acceptance: ' + ($rows.Count - $bad.Count) + '/' + $rows.Count + ' arms as expected')
if (@(Get-Process -Name ninfer-serve -ErrorAction SilentlyContinue).Count -ne 0) {
    Write-Host 'warning: a ninfer-serve process is still running'
}
if ($bad.Count -ne 0) { Write-Output 'SKEW_ACCEPTANCE_FAILED'; exit 1 }
Write-Output 'SKEW_ACCEPTANCE_DONE'
