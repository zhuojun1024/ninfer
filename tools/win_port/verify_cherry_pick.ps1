# End-to-end verification for the upstream cherry-pick campaign (Tier 1 + Tier 2).
#
# Run AFTER restarting the 3456 service with the new binary (build-win2/apps/ninfer-serve.exe).
# This script never starts or stops the service; it only measures the running one.
#
#   pwsh -File tools/win_port/verify_cherry_pick.ps1
#
# It runs bench_serve.ps1 against the running service and writes the results to
# profiles/bench/upstream_cherry_pick/. Compare the numbers against the pre-cherry-pick
# baseline (decode ~56.9 tok/s, prefill ~1,588 tok/s on the dual 5060 Ti).
[CmdletBinding()]
param(
    [string] $BaseUrl = "http://127.0.0.1:3456/v1",
    [string] $Model   = "",
    [int]    $DecodeTokens = 256,
    [int[]]  $PrefillSizes = @(2048, 8192, 32768)
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$outDir = Join-Path $root "profiles/bench/upstream_cherry_pick"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$outFile = Join-Path $outDir "bench-new.json"

Write-Host "=== Cherry-pick e2e verification ==="
Write-Host "BaseUrl : $BaseUrl"
Write-Host "OutFile : $outFile"
Write-Host ""

try {
    $models = Invoke-RestMethod -Uri "$BaseUrl/models" -Method Get -TimeoutSec 15
    Write-Host "[ok] service is up"
} catch {
    Write-Error "service not reachable at $BaseUrl/models : $_"
    exit 1
}

$bench = Join-Path $root "tools/win_port/bench_serve.ps1"
& $bench -BaseUrl $BaseUrl -Label "cherry-pick-new" -Model $Model -OutFile $outFile -DecodeTokens $DecodeTokens -PrefillSizes $PrefillSizes
if ($LASTEXITCODE -ne 0) {
    Write-Error "bench_serve.ps1 failed (exit $LASTEXITCODE)"
    exit 1
}

Write-Host ""
Write-Host "=== Summary (new binary) ==="
Get-Content $outFile | ConvertFrom-Json | ForEach-Object {
    $r = $_
    if ($r.tag -like "prefill*") {
        Write-Host ("  prefill  {0} tok : {1} tok/s" -f $r.prompt_n, $r.prompt_per_second)
    } elseif ($r.tag -like "decode*") {
        Write-Host ("  decode   {0} tok : {1} tok/s" -f $r.predicted_n, $r.predicted_per_second)
    }
}
Write-Host ""
Write-Host "Baseline (pre-cherry-pick, dual 5060 Ti): decode ~56.9 tok/s, prefill ~1,588 tok/s."
Write-Host "Results written to: $outFile"

