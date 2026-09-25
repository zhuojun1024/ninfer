# Stall-injection sweep, retargeted past warmup.
#
# The watchdog dump taken at the first idle after warmup shows calls=268 (in-kernel collectives) and
# the last collective at base+(129) with six slices, so warmup owns collectives 1..268 and a real
# request starts at 269 with roughly 130-180 collectives per DFlash2 round. Injecting inside that
# range places a bounded-spin give-up in the middle of a prefill or a verify window instead of in the
# warmup, which is what decides whether a give-up stays a clean 503 or feeds the round's consumers
# with un-updated partial sums.
#
#   pwsh -File tools/tp_bootstrap/r72_stall_injection.ps1
[CmdletBinding()]
param(
    [string] $Model = 'D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_final.ninfer',
    [int] $DraftTokens = 5,
    [int] $Port = 8099,
    [string] $OutDir = 'build-win/r72',
    [int[]] $Calls = @(280, 300, 330, 360, 400, 440, 480, 560, 700),
    [int] $Requests = 2
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$prompts = @(
    @{ name = 'reason'; text = 'A train leaves Station A at 09:15 travelling 72 km/h, and a second train leaves Station B at 09:40 travelling 96 km/h. The stations are 310 km apart. Solve this step by step, showing every calculation and checking the result.' },
    @{ name = 'prose';  text = 'Write a detailed description of how a submarine works, about 400 words.' }
)

$rows = @()
foreach ($call in $Calls) {
    Get-Process -Name ninfer-serve -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 3
    $log = Join-Path $OutDir ("inject-" + $call + ".log")
    $env:NINFER_TP2_AR_FAULT_SKIP_PEER_CALL = "$call"
    # Default the diagnostics on (the watchdog is on in production too), but let an explicit outer
    # setting win so NINFER_TP2_AR_WATCHDOG=0 can be used to check the off switch.
    if (-not $env:NINFER_TP2_AR_WATCHDOG) { $env:NINFER_TP2_AR_WATCHDOG = "1" }
    $server = Start-Process -FilePath "pwsh" -PassThru -WindowStyle Hidden -ArgumentList @(
        "-NoProfile", "-File", (Join-Path $repo "tools/win_port/serve.ps1"),
        "-Model", $Model, "-Spec", "dflash2", "-DraftTokens", "$DraftTokens",
        "-Port", "$Port", "-LogFile", $log
    )
    Remove-Item Env:NINFER_TP2_AR_FAULT_SKIP_PEER_CALL -ErrorAction SilentlyContinue
    Remove-Item Env:NINFER_TP2_AR_WATCHDOG -ErrorAction SilentlyContinue
    $statuses = @()
    try {
        $deadline = (Get-Date).AddMinutes(4)
        $ready = $false
        while ((Get-Date) -lt $deadline) {
            Start-Sleep -Seconds 3
            if (@(Get-Process -Name ninfer-serve -ErrorAction SilentlyContinue).Count -eq 0) { break }
            try { if ((Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5).Content) { $ready = $true; break } } catch { }
        }
        if (-not $ready) { $statuses += 'no-start' } else {
            $modelId = (Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/models" -TimeoutSec 30).data[0].id
            for ($i = 0; $i -lt $Requests; $i++) {
                if (@(Get-Process -Name ninfer-serve -ErrorAction SilentlyContinue).Count -eq 0) { $statuses += 'process-gone'; break }
                $payload = @{
                    model = $modelId
                    messages = @(@{ role = 'user'; content = $prompts[$i % 2].text })
                    max_tokens = 256; temperature = 0.7; top_k = 20; top_p = 0.8; stream = $false
                } | ConvertTo-Json -Depth 8 -Compress
                try {
                    $null = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post `
                        -ContentType 'application/json; charset=utf-8' `
                        -Body ([System.Text.Encoding]::UTF8.GetBytes($payload)) -TimeoutSec 300
                    $statuses += 'ok'
                } catch {
                    $msg = ($_.ErrorDetails.Message + ' ' + $_.Exception.Message)
                    if ($msg -match 'service_unavailable' -or $msg -match 'stalled') { $statuses += '503' }
                    elseif ($msg -match 'actively refused|Unable to connect') { $statuses += 'conn-refused' }
                    else { $statuses += ('err:' + $msg.Substring(0, [Math]::Min(90, $msg.Length))) }
                }
            }
        }
    } catch {
        $statuses += ('harness:' + $_.Exception.Message.Substring(0, [Math]::Min(60, $_.Exception.Message.Length)))
    } finally {
        $illegal = $false
        $warmupFatal = $false
        if (Test-Path $log) {
            $illegal = [bool](Select-String -Path $log -Pattern 'IllegalAddress|illegal memory' -Quiet)
            $warmupFatal = [bool](Select-String -Path $log -Pattern 'FATAL warmup failed' -Quiet)
        }
        Write-Host ("INJECT call=" + $call + " -> " + ($statuses -join ',') + " illegal=" + $illegal + " warmupFatal=" + $warmupFatal)
        $rows += [pscustomobject]@{ call = $call; statuses = ($statuses -join ','); illegal = $illegal; warmupFatal = $warmupFatal }
        Get-Process -Name ninfer-serve -ErrorAction SilentlyContinue | Stop-Process -Force
        if ($server -and -not $server.HasExited) { $server | Stop-Process -Force }
        Start-Sleep -Seconds 3
    }
}
$rows | ConvertTo-Json | Set-Content (Join-Path $OutDir 'injection-summary.json') -Encoding UTF8
Write-Output 'INJECTION_DONE'
