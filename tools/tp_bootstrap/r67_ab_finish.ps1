# Finish the r67 A/B after the q4all sampling K=7 arm hit an engine CUDA fault.
#
#   pwsh -File tools/tp_bootstrap/r67_ab_finish.ps1
#
# Each step is isolated: a failure is recorded and the remaining steps still run, so one crash does
# not discard the suites.
$ErrorActionPreference = 'Stop'
Set-Location 'D:\Documents\workbench\ninfer'

$base = 'D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_q4all.ninfer'
$treat = 'D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_qkv4.ninfer'
$out = 'build-win/r67'
New-Item -ItemType Directory -Force -Path $out | Out-Null

function Invoke-Step([string] $name, [scriptblock] $body) {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $code = 0
    try { & $body; $code = $LASTEXITCODE } catch { Write-Host ("STEP {0} THREW: {1}" -f $name, $_); $code = 99 }
    Write-Host ("STEP {0} exit={1} elapsed_s={2}" -f $name, $code, [math]::Round($sw.Elapsed.TotalSeconds, 1))
}

Invoke-Step 'sample-q4all-k7-retry' {
    & pwsh -NoProfile -File 'tools/tp_bootstrap/r62_sampling_ab.ps1' -Tag 'r67-q4all-k7' -Model $base -DraftTokens 7 -Port 8099 -OutDir $out -Reps 5
}
Invoke-Step 'sample-q4all-k5' {
    & pwsh -NoProfile -File 'tools/tp_bootstrap/r62_sampling_ab.ps1' -Tag 'r67-q4all-k5' -Model $base -DraftTokens 5 -Port 8099 -OutDir $out -Reps 5
}
Invoke-Step 'suites-qkv4' {
    & pwsh -NoProfile -File 'tools/tp_bootstrap/r66_suites.ps1' -Model $treat -OutDir "$out/suites-qkv4"
}
Invoke-Step 'suites-q4all' {
    & pwsh -NoProfile -File 'tools/tp_bootstrap/r66_suites.ps1' -Model $base -OutDir "$out/suites-q4all"
}
Write-Host 'R67_AB_FINISH_DONE'
