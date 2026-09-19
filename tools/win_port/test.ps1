# Run the native Windows test suite through ctest.
#
#   pwsh -File tools/win_port/test.ps1                       # whole suite
#   pwsh -File tools/win_port/test.ps1 -Filter "artifact"    # one regex of test names
#   pwsh -File tools/win_port/test.ps1 -Build                # build the suite first
#
# The test executables load avcodec/swscale/libcurl at run time, so both DLL directories must be on PATH
# before ctest launches anything. Without that the Windows loader raises a missing-DLL dialog for every
# executable it starts, which floods the desktop instead of reporting a test failure.
[CmdletBinding()]
param(
    [string] $BuildDir = "",
    [string] $Filter = "",
    [switch] $Build,
    [int] $Jobs = 1,
    [string] $FfmpegRoot = "D:/ffmpeg-dev/expanded/ffmpeg-master-latest-win64-gpl-shared",
    [string] $LibcurlRoot = "D:/curl-dev/expanded/curl-8.22.0_1-win64-mingw"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $BuildDir) { $BuildDir = Join-Path $root "build-win" }
if (-not (Test-Path (Join-Path $BuildDir "CTestTestfile.cmake"))) {
    throw "no configured test suite in $BuildDir; run: tools/win_port/build.ps1 -Configure -Tests 1"
}

if ($Build) { & (Join-Path $PSScriptRoot "build.ps1") -Tests 1 }

$env:PATH = "$FfmpegRoot/bin;$LibcurlRoot/bin;$env:PATH"

$arguments = @("--output-on-failure", "--timeout", "300")
if ($Jobs -gt 1) { $arguments += @("-j", "$Jobs") }
if ($Filter) { $arguments += @("-R", $Filter) }

Push-Location $BuildDir
try {
    & ctest @arguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
