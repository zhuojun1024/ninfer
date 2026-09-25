# Higher-power r68 A/B: 30 sampling repetitions per arm (the 15-rep run left the K=5 signal at
# ~2.4 sigma). Same two artifacts, same serialization rules.
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

foreach ($k in @(7, 5)) {
    Invoke-Step ("hi-cb4-k" + $k) {
        & pwsh -NoProfile -File 'tools/tp_bootstrap/r62_sampling_ab.ps1' -Tag ("r68hi-cb4-k" + $k) -Model $treat -DraftTokens $k -Port 8099 -OutDir $out -Reps 30
    }
    Invoke-Step ("hi-q4all-k" + $k) {
        & pwsh -NoProfile -File 'tools/tp_bootstrap/r62_sampling_ab.ps1' -Tag ("r68hi-q4all-k" + $k) -Model $base -DraftTokens $k -Port 8099 -OutDir $out -Reps 30
    }
}
Write-Host 'R68_HI_DONE'
