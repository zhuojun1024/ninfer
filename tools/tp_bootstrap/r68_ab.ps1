# r68 acceptance A/B and suites for the selector-codebook lever (PLAN.md section 3.9 lever 1).
#
#   pwsh -File tools/tp_bootstrap/r68_ab.ps1
#
# Treatment = the cb4 artifact (r66 q4 draft + Q4 selector codebooks); baseline = the r66 q4all
# artifact, re-measured on the same binary. Serialized: every step holds both cards and the serve
# steps bind port 8099. Steps are isolated so one crash does not discard the rest.
$ErrorActionPreference = 'Stop'
Set-Location 'D:\Documents\workbench\ninfer'

$base = 'D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_q4all.ninfer'
$treat = 'D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_cb4.ninfer'
$out = 'build-win/r68'
New-Item -ItemType Directory -Force -Path $out | Out-Null

function Invoke-Step([string] $name, [scriptblock] $body) {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $code = 0
    try { & $body; $code = $LASTEXITCODE } catch { Write-Host ("STEP {0} THREW: {1}" -f $name, $_); $code = 99 }
    Write-Host ("STEP {0} exit={1} elapsed_s={2}" -f $name, $code, [math]::Round($sw.Elapsed.TotalSeconds, 1))
}

Invoke-Step 'sample-cb4-k7' {
    & pwsh -NoProfile -File 'tools/tp_bootstrap/r62_sampling_ab.ps1' -Tag 'r68-cb4-k7' -Model $treat -DraftTokens 7 -Port 8099 -OutDir $out -Reps 5
}
Invoke-Step 'sample-cb4-k5' {
    & pwsh -NoProfile -File 'tools/tp_bootstrap/r62_sampling_ab.ps1' -Tag 'r68-cb4-k5' -Model $treat -DraftTokens 5 -Port 8099 -OutDir $out -Reps 5
}
Invoke-Step 'sample-q4all-k7' {
    & pwsh -NoProfile -File 'tools/tp_bootstrap/r62_sampling_ab.ps1' -Tag 'r68-q4all-k7' -Model $base -DraftTokens 7 -Port 8099 -OutDir $out -Reps 5
}
Invoke-Step 'sample-q4all-k5' {
    & pwsh -NoProfile -File 'tools/tp_bootstrap/r62_sampling_ab.ps1' -Tag 'r68-q4all-k5' -Model $base -DraftTokens 5 -Port 8099 -OutDir $out -Reps 5
}
Invoke-Step 'suites-cb4' {
    $env:NINFER_TEST_ROUTE = 'dflash2'
    & pwsh -NoProfile -File 'tools/tp_bootstrap/r66_suites.ps1' -Model $treat -OutDir "$out/suites-cb4"
    Remove-Item Env:NINFER_TEST_ROUTE -ErrorAction SilentlyContinue
}
Write-Host 'R68_AB_DONE'
