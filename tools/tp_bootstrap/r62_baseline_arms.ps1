# Step 0 of the §8 campaign (PLAN.md): reproduce the §3.7 r57 acceptance numbers on the current
# binary, at K=7 and K=5, before any op development starts.
#
#   pwsh -File tools/tp_bootstrap/r62_baseline_arms.ps1
#
# One engine at a time (the r54 arm script enforces it). Baseline artifact = the r57 draft-all piece
# that PLAN.md §3.7 recorded as K=7 acceptance 25.3% (706/2791) / 56.9 tok/s.
[CmdletBinding()]
param(
    [string] $Model = "D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_draftall.ninfer",
    [string] $OutDir = "",
    [int[]] $Ks = @(7, 5)
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $OutDir) { $OutDir = Join-Path $repo "build-win/r62" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$arm = Join-Path $repo "tools/tp_bootstrap/r54_dflash2_arm.ps1"

if (-not (Test-Path $Model)) { throw "baseline artifact not found: $Model" }

foreach ($k in $Ks) {
    & pwsh -NoProfile -File $arm -Tag ("base-k" + $k) -Model $Model -DraftTokens $k -OutDir $OutDir
    if ($LASTEXITCODE -ne 0) { throw ("baseline arm K=" + $k + " failed with exit " + $LASTEXITCODE) }
}
Write-Host "both baseline arms done"
