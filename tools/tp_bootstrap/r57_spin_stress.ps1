# Stress the in-kernel AR transport at op level to reproduce the probabilistic spin hang in minimal
# form: the device-pair test's 200-deep queued allreduce queues maximize the two devices' token
# counter overlap per unit time. Each run is bounded by -TimeoutSec; a run that does not exit is a
# hang and keeps its [ar-watch] tail (NINFER_TP2_AR_WATCHDOG=1) for classification.
#
#   pwsh -File tools/tp_bootstrap/r57_spin_stress.ps1 -Runs 40
[CmdletBinding()]
param(
    [int] $Runs = 40,
    [int] $TimeoutSec = 60,
    [string] $Exe = "build-win/tests/ninfer_tp_device_pair_test.exe",
    [string] $OutDir = "build-win/r57"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$env:NINFER_TP2_AR_WATCHDOG = "1"

$hangs = 0
for ($i = 1; $i -le $Runs; $i++) {
    $out = Join-Path $OutDir ("stress-" + $i + ".log")
    $err = Join-Path $OutDir ("stress-" + $i + ".err")
    $p = Start-Process -FilePath (Join-Path $repo $Exe) -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    while (-not $p.HasExited -and $sw.Elapsed.TotalSeconds -lt $TimeoutSec) {
        Start-Sleep -Seconds 2
    }
    if ($p.HasExited) {
        Write-Host ("run {0}: exit {1} in {2:n0}s" -f $i, $p.ExitCode, $sw.Elapsed.TotalSeconds)
    } else {
        $hangs++
        Write-Host ("run {0}: HANG (still running after {1}s)" -f $i, $TimeoutSec)
        Copy-Item -LiteralPath $err (Join-Path $OutDir ("hang-stress-" + $i + ".err")) -Force
        Copy-Item -LiteralPath $out (Join-Path $OutDir ("hang-stress-" + $i + ".log")) -Force
        Get-Content -LiteralPath $err -Tail 25 | Set-Content (Join-Path $OutDir ("hang-stress-" + $i + "-tail.txt"))
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader |
            Out-File (Join-Path $OutDir ("hang-stress-" + $i + "-gpu.txt"))
        $p | Stop-Process -Force
    }
    Start-Sleep -Milliseconds 500
}
Write-Host ("done: {0} runs, {1} hangs" -f $Runs, $hangs)
