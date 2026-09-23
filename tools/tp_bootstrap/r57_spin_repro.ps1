# Reproduce the probabilistic TP-2 in-kernel AR spin hang: one fresh server + one greedy probe per
# attempt with the [ar-watch] state probe enabled (NINFER_TP2_AR_WATCHDOG=1). A hang is a probe that
# does not finish within -TimeoutSec; that attempt saves the AR state tail and a GPU snapshot.
#
#   pwsh -File tools/tp_bootstrap/r57_spin_repro.ps1 -Attempts 8
#
# Exactly one model process machine-wide: each attempt stops the previous server first.
[CmdletBinding()]
param(
    [int] $Attempts = 8,
    [int] $TimeoutSec = 90,
    [int] $Port = 8099,
    [int] $DraftTokens = 7,
    [string] $Model = "D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2.ninfer",
    [string] $Model2 = "",
    [string] $OutDir = "build-win/r57"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$env:NINFER_TP2_TIMING = "1"
$env:NINFER_TP2_AR_WATCHDOG = "1"

$hangs = 0
for ($i = 1; $i -le $Attempts; $i++) {
    Get-Process -Name ninfer-serve -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 3
    $log = Join-Path $OutDir ("serve-" + $i + ".log")
    $jsonl = Join-Path $OutDir ("probe-" + $i + ".jsonl")

    $arm_model = if ($Model2 -and ($i % 2 -eq 0)) { $Model2 } else { $Model }
    Write-Host ("attempt {0}: model {1}" -f $i, $arm_model)
    $server = Start-Process -FilePath "pwsh" -PassThru -WindowStyle Hidden -ArgumentList @(
        "-NoProfile", "-File", (Join-Path $repo "tools/win_port/serve.ps1"),
        "-Model", $arm_model, "-Spec", "dflash2", "-DraftTokens", "$DraftTokens",
        "-Port", "$Port", "-LogFile", $log
    )
    $deadline = (Get-Date).AddMinutes(8)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 5
        try {
            $health = (Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5).Content
            if ($health) { $ready = $true; break }
        } catch { }
    }
    if (-not $ready) {
        Write-Host ("attempt {0}: server never became ready; see {1}" -f $i, $log)
        Get-Process -Name ninfer-serve -ErrorAction SilentlyContinue | Stop-Process -Force
        continue
    }

    $probe = Start-Process -FilePath "pwsh" -PassThru -WindowStyle Hidden -ArgumentList @(
        "-NoProfile", "-File", (Join-Path $repo "tools/win_port/greedy_probe.ps1"),
        "-BaseUrl", "http://127.0.0.1:$Port/v1", "-MaxTokens", "160", "-OutFile", $jsonl
    )
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    while (-not $probe.HasExited -and $sw.Elapsed.TotalSeconds -lt $TimeoutSec) {
        Start-Sleep -Seconds 5
    }
    if ($probe.HasExited) {
        Write-Host ("attempt {0}: OK in {1:n0}s" -f $i, $sw.Elapsed.TotalSeconds)
    } else {
        $hangs++
        Write-Host ("attempt {0}: HANG (probe still running after {1}s)" -f $i, $TimeoutSec)
        $gpu = Join-Path $OutDir ("hang-" + $i + "-gpu.txt")
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | Out-File $gpu
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | Out-File -Append $gpu
        Copy-Item -LiteralPath $log (Join-Path $OutDir ("hang-" + $i + "-serve.log")) -Force
        Get-Content -LiteralPath $log -Tail 60 | Set-Content (Join-Path $OutDir ("hang-" + $i + "-tail.txt"))
    }
    $probe | Stop-Process -Force -ErrorAction SilentlyContinue
    Get-Process -Name ninfer-serve -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 2
}
Write-Host ("done: {0} attempts, {1} hangs" -f $Attempts, $hangs)
