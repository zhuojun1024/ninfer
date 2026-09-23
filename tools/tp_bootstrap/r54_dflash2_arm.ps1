# One DFlash2 acceptance arm: start one engine, probe it greedily, stop it, summarise.
#
#   pwsh -File tools/tp_bootstrap/r54_dflash2_arm.ps1 -Tag base -Model D:/LLM/baseline.ninfer -DraftTokens 7
#
# Run exactly one arm at a time: the two-request CLI holds both cards.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $Tag,
    [Parameter(Mandatory = $true)][string] $Model,
    [int] $DraftTokens = 7,
    [int] $Port = 8099,
    [string] $OutDir = "build-win/r54",
    [int] $MaxTokens = 160
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$log = Join-Path $OutDir ("serve-" + $Tag + ".log")
$jsonl = Join-Path $OutDir ($Tag + "-k" + $DraftTokens + ".jsonl")

Get-Process -Name ninfer-serve -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 3

$env:NINFER_TP2_TIMING = "1"
$server = Start-Process -FilePath "pwsh" -PassThru -WindowStyle Hidden -ArgumentList @(
    "-NoProfile", "-File", (Join-Path $repo "tools/win_port/serve.ps1"),
    "-Model", $Model, "-Spec", "dflash2", "-DraftTokens", "$DraftTokens",
    "-Port", "$Port", "-LogFile", $log
)
try {
    $deadline = (Get-Date).AddMinutes(8)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 5
        try {
            $health = (Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5).Content
            if ($health) { $ready = $true; break }
        } catch { }
    }
    if (-not $ready) { throw "server did not become ready; see $log" }
    Write-Host ("server ready: " + $Tag)

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & pwsh -NoProfile -File (Join-Path $repo "tools/win_port/greedy_probe.ps1") `
        -BaseUrl "http://127.0.0.1:$Port/v1" -MaxTokens $MaxTokens -OutFile $jsonl
    $elapsed = $sw.Elapsed.TotalSeconds

    $rows = Get-Content -LiteralPath $jsonl | ForEach-Object { $_ | ConvertFrom-Json }
    $drafted = ($rows | Measure-Object -Property draft_n -Sum).Sum
    $accepted = ($rows | Measure-Object -Property draft_n_accepted -Sum).Sum
    $tokens = ($rows | Measure-Object -Property tokens -Sum).Sum
    $rate = if ($drafted -gt 0) { 100.0 * $accepted / $drafted } else { 0.0 }
    Write-Host ("SUMMARY tag=" + $Tag + " k=" + $DraftTokens +
        " drafted=" + $drafted + " accepted=" + $accepted +
        " acceptance=" + [math]::Round($rate, 2) + "%" +
        " tokens=" + $tokens + " elapsed_s=" + [math]::Round($elapsed, 1) +
        " tok_per_s=" + [math]::Round($tokens / $elapsed, 1))
} finally {
    Get-Process -Name ninfer-serve -ErrorAction SilentlyContinue | Stop-Process -Force
    if ($server -and -not $server.HasExited) { $server | Stop-Process -Force }
    Start-Sleep -Seconds 3
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    Write-Host ("arm done: " + $Tag + " log " + $log)
}
