# Long-run stress for the rare TP-2 transport failures (one cudaErrorIllegalAddress, one bounded-spin
# 503) seen in the section 3.9 acceptance arms. Unlike the A/B harness this one keeps going after a
# failed request: the bounded-spin design fails the request and recovers, so one session can collect
# several events. The transport watchdog (NINFER_TP2_AR_WATCHDOG=1) dumps the arrival/order slots of
# both devices when the host-side collective counter stops advancing, which is what decides whether a
# stall is a token mismatch (protocol) or a missing peer write (lost store or a slow device).
#
#   pwsh -File tools/tp_bootstrap/r71_ar_stress.ps1 -Tag r71-stress -DraftTokens 5 -Requests 600
[CmdletBinding()]
param(
    [string] $Tag = 'r71-stress',
    [string] $Model = 'D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_final.ninfer',
    [int] $DraftTokens = 5,
    [int] $Port = 8099,
    [string] $OutDir = 'build-win/r71',
    [int] $Requests = 600,
    [int] $MaxTokens = 256,
    [int] $TimeoutSec = 900
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $PSScriptRoot "ar_watch.ps1")
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$log = Join-Path $OutDir ("stress-" + $Tag + ".log")
$jsonl = Join-Path $OutDir ("stress-" + $Tag + ".jsonl")

Get-Process -Name ninfer-serve -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 3

# The watchdog writes to stderr, which serve.ps1 routes to the log file.
$env:NINFER_TP2_AR_WATCHDOG = '1'
$server = Start-Process -FilePath "pwsh" -PassThru -WindowStyle Hidden -ArgumentList @(
    "-NoProfile", "-File", (Join-Path $repo "tools/win_port/serve.ps1"),
    "-Model", $Model, "-Spec", "dflash2", "-DraftTokens", "$DraftTokens",
    "-Port", "$Port", "-LogFile", $log
)

$failures = 0
$ok = 0
function Serve-Alive { @(Get-Process -Name ninfer-serve -ErrorAction SilentlyContinue).Count -ne 0 }

try {
    $deadline = (Get-Date).AddMinutes(8)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 5
        try { if ((Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5).Content) { $ready = $true; break } } catch { }
    }
    if (-not $ready) { throw "server did not become ready; see $log" }
    $modelId = (Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/models" -TimeoutSec 30).data[0].id
    Write-Host ("server ready: " + $Tag + " watchdog=" + $env:NINFER_TP2_AR_WATCHDOG)

    $prompts = @(
        @{ name = 'reason'; text = 'A train leaves Station A at 09:15 travelling 72 km/h, and a second train leaves Station B at 09:40 travelling 96 km/h. The stations are 310 km apart. Solve this step by step, showing every calculation and checking the result.' },
        @{ name = 'prose';  text = 'Write a detailed description of how a submarine works, about 400 words.' },
        @{ name = 'code';   text = 'Write a Python function that parses a CSV file with quoted fields, including tests. Explain the edge cases.' }
    )

    $lines = @()
    for ($i = 1; $i -le $Requests; $i++) {
        $prompt = $prompts[($i - 1) % $prompts.Count]
        $payload = @{
            model       = $modelId
            messages    = @(@{ role = 'user'; content = $prompt.text })
            max_tokens  = $MaxTokens
            temperature = 0.7
            top_k       = 20
            top_p       = 0.8
            stream      = $false
        } | ConvertTo-Json -Depth 8 -Compress
        $body = [System.Text.Encoding]::UTF8.GetBytes($payload)
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $status = 'ok'
        $detail = ''
        try {
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post `
                -ContentType "application/json; charset=utf-8" -Body $body -TimeoutSec $TimeoutSec
            $ok++
        } catch {
            $status = 'failed'
            $failures++
            $detail = ($_.ErrorDetails.Message + ' ' + $_.Exception.Message)
            $detail = $detail.Substring(0, [Math]::Min(600, $detail.Length))
            Write-Host ("REQUEST " + $i + " FAILED: " + $detail)
            # Keep the transport state of this failure under a name the next arm cannot overwrite.
            Save-ArWatchEvidence -LogPath $log -Stem (Join-Path $OutDir ("stress-" + $Tag + "-req" + $i)) | Out-Null
        }
        $lines += (@{ n = $i; prompt = $prompt.name; status = $status; ms = [math]::Round($sw.Elapsed.TotalMilliseconds, 0); detail = $detail } | ConvertTo-Json -Compress)
        if ($i % 25 -eq 0) {
            Write-Host ("progress " + $i + "/" + $Requests + " ok=" + $ok + " failed=" + $failures)
            Set-Content -Path $jsonl -Value $lines -Encoding UTF8
        }
        if (-not (Serve-Alive)) {
            Write-Host ("SERVER PROCESS GONE after request " + $i)
            break
        }
    }
    Set-Content -Path $jsonl -Value $lines -Encoding UTF8
    Write-Host ("STRESS_DONE ok=" + $ok + " failed=" + $failures)
} finally {
    Get-Process -Name ninfer-serve -ErrorAction SilentlyContinue | Stop-Process -Force
    if ($server -and -not $server.HasExited) { $server | Stop-Process -Force }
    Remove-Item Env:NINFER_TP2_AR_WATCHDOG -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    Write-Host ("stress arm done: " + $Tag + " log " + $log)
}
