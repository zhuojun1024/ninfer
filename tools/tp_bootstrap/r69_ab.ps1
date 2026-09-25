# r69 acceptance A/B and suites for the final promoted artifact (official recipe carrying the
# r62-r66 draft Q4 set plus the r67 fused-QKV and r68 codebook levers).
#
#   pwsh -File tools/tp_bootstrap/r69_ab.ps1
#
# Baseline = the r66 q4all artifact, which differs from the treatment only by those two levers.
# 30 sampling repetitions per arm (the 15-rep arm-to-arm noise floor is ~2 pp). Serialized.
$ErrorActionPreference = 'Stop'
Set-Location 'D:\Documents\workbench\ninfer'

$base = 'D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_q4all.ninfer'
$treat = 'D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_final.ninfer'
$out = 'build-win/r69'
New-Item -ItemType Directory -Force -Path $out | Out-Null

function Invoke-Step([string] $name, [scriptblock] $body) {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $code = 0
    try { & $body; $code = $LASTEXITCODE } catch { Write-Host ("STEP {0} THREW: {1}" -f $name, $_); $code = 99 }
    Write-Host ("STEP {0} exit={1} elapsed_s={2}" -f $name, $code, [math]::Round($sw.Elapsed.TotalSeconds, 1))
}

foreach ($k in @(7, 5)) {
    Invoke-Step ("hi-final-k" + $k) {
        & pwsh -NoProfile -File 'tools/tp_bootstrap/r62_sampling_ab.ps1' -Tag ("r69-final-k" + $k) -Model $treat -DraftTokens $k -Port 8099 -OutDir $out -Reps 30
    }
    Invoke-Step ("hi-q4all-k" + $k) {
        & pwsh -NoProfile -File 'tools/tp_bootstrap/r62_sampling_ab.ps1' -Tag ("r69-q4all-k" + $k) -Model $base -DraftTokens $k -Port 8099 -OutDir $out -Reps 30
    }
}
Invoke-Step 'suites-final' {
    $env:NINFER_TEST_ROUTE = 'dflash2'
    & pwsh -NoProfile -File 'tools/tp_bootstrap/r66_suites.ps1' -Model $treat -OutDir "$out/suites-final"
    Remove-Item Env:NINFER_TEST_ROUTE -ErrorAction SilentlyContinue
}
Write-Host 'R69_AB_DONE'
