# r67 acceptance A/B and suite re-baseline for the section 3.9 QKV->q4 lever.
#
#   pwsh -File tools/tp_bootstrap/r67_ab.ps1
#
# Serialized on purpose: every step holds both cards and the serve steps bind port 8099.
# Baseline arm = the r66 q4all artifact (same code path as the stock Q8 draft); treatment arm =
# the r67 qkv4 artifact. Both run on the same rebuilt binary.
$ErrorActionPreference = 'Stop'
Set-Location 'D:\Documents\workbench\ninfer'

$base = 'D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_q4all.ninfer'
$treat = 'D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_qkv4.ninfer'
$out = 'build-win/r67'
New-Item -ItemType Directory -Force -Path $out | Out-Null

Write-Host '=== step arms: r67-qkv4 ==='
& pwsh -NoProfile -File 'tools/tp_bootstrap/r62_step_arms.ps1' -Tag 'r67-qkv4' -Model $treat -OutDir $out
if ($LASTEXITCODE -ne 0) { throw 'step arms failed: r67-qkv4' }

Write-Host '=== step arms: r67-q4all baseline ==='
& pwsh -NoProfile -File 'tools/tp_bootstrap/r62_step_arms.ps1' -Tag 'r67-q4all' -Model $base -OutDir $out
if ($LASTEXITCODE -ne 0) { throw 'step arms failed: r67-q4all' }

Write-Host '=== suites: qkv4 ==='
& pwsh -NoProfile -File 'tools/tp_bootstrap/r66_suites.ps1' -Model $treat -OutDir "$out/suites-qkv4"
if ($LASTEXITCODE -ne 0) { throw 'suites failed: qkv4' }

Write-Host '=== suites: q4all ==='
& pwsh -NoProfile -File 'tools/tp_bootstrap/r66_suites.ps1' -Model $base -OutDir "$out/suites-q4all"
if ($LASTEXITCODE -ne 0) { throw 'suites failed: q4all' }

Write-Host 'R67_AB_DONE'
