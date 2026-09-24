# One campaign step's acceptance A/B: the greedy 7x160 probe at K=7/K=5 plus the sampling A/B.
#
#   pwsh -File tools/tp_bootstrap/r62_step_arms.ps1 -Tag finish4 -Model D:/LLM/<artifact>.ninfer
#
# Run exactly one engine at a time; the two scripts this drives both enforce that.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $Tag,
    [Parameter(Mandatory = $true)][string] $Model,
    [int] $SamplingReps = 5,
    [int] $Port = 8099,
    [string] $OutDir = "build-win/r62"
)

$ErrorActionPreference = "Stop"
$arm = Join-Path $PSScriptRoot "r54_dflash2_arm.ps1"
$sampling = Join-Path $PSScriptRoot "r62_sampling_ab.ps1"

foreach ($k in @(7, 5)) {
    & pwsh -NoProfile -File $arm -Tag ($Tag + "-k" + $k) -Model $Model -DraftTokens $k -Port $Port -OutDir $OutDir
    if ($LASTEXITCODE -ne 0) { throw ("greedy K=" + $k + " failed for " + $Tag) }
}
foreach ($k in @(7, 5)) {
    & pwsh -NoProfile -File $sampling -Tag ("sample-" + $Tag + "-k" + $k) -Model $Model -DraftTokens $k -Port $Port -OutDir $OutDir -Reps $SamplingReps
    if ($LASTEXITCODE -ne 0) { throw ("sampling K=" + $k + " failed for " + $Tag) }
}
Write-Host ("step arms done: " + $Tag)
