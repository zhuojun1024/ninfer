# r66 three-suite re-baseline on the merged Q4 draft artifact (PLAN.md section 8, step 5).
#
#   pwsh -File tools/tp_bootstrap/r66_suites.ps1 -Model D:/LLM/<artifact>.ninfer
#
# The suites are self-consistency checks (bit-identical repeats, cross-process digests, retention
# assertions), not absolute goldens, so they re-baseline on a new artifact without editing them.
# Each one loads both cards; run nothing else while this is up.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $Model,
    [string] $OutDir = "build-win/r66"
)

$ErrorActionPreference = 'Stop'
Set-Location 'D:\Documents\workbench\ninfer'
# The Engine pulls the media paths in, so the FFmpeg and libcurl runtime DLLs must be resolvable
# or every suite dies at load with 0xC0000135 (PLAN.md environment table).
$env:PATH = 'D:/ffmpeg-dev/expanded/ffmpeg-master-latest-win64-gpl-shared/bin;' +
            'D:/curl-dev/expanded/curl-8.22.0_1-win64-mingw/bin;' + $env:PATH
if (-not (Test-Path $Model)) { throw "artifact not found: $Model" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$env:NINFER_TEST_ARTIFACT = $Model

$results = @()
function Invoke-Suite([string] $name, [string] $exe, [string[]] $extra) {
    $log = Join-Path $OutDir ($name + ".log")
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $exe @extra *> $log
    $code = $LASTEXITCODE
    $elapsed = [math]::Round($sw.Elapsed.TotalSeconds, 1)
    "SUITE $name exit=$code elapsed_s=$elapsed log=$log"
    Get-Content $log | Select-Object -Last 4 | ForEach-Object { "  | $_" }
    $script:results += [pscustomobject]@{ suite = $name; exit = $code; elapsed_s = $elapsed }
}

Invoke-Suite "append-k7" "build-win/tests/ninfer_qwen3_5_tp2_dflash_append_test.exe" @("--k", "7")
Invoke-Suite "append-k5" "build-win/tests/ninfer_qwen3_5_tp2_dflash_append_test.exe" @("--k", "5")
Invoke-Suite "solo-a" "build-win/tests/ninfer_qwen3_5_tp2_dflash_solo_test.exe" @()
Invoke-Suite "solo-b" "build-win/tests/ninfer_qwen3_5_tp2_dflash_solo_test.exe" @()
Invoke-Suite "sessions" "build-win/tests/ninfer_qwen3_5_tp2_sessions_test.exe" @()

"--- digest lines ---"
foreach ($name in @("solo-a", "solo-b", "sessions", "append-k7", "append-k5")) {
    $lines = Select-String -Path (Join-Path $OutDir ($name + ".log")) -Pattern 'digest|fnv1a|passed' -ErrorAction SilentlyContinue
    foreach ($line in $lines) { "{0}: {1}" -f $name, $line.Line.Trim() }
}

$bad = @($results | Where-Object { $_.exit -ne 0 })
"SUITES_DONE failures=$($bad.Count)"
$results | Format-Table -AutoSize | Out-String
