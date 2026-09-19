# Configure and build NInfer natively on Windows.
#
#   pwsh -File tools/win_port/build.ps1                            # incremental build
#   pwsh -File tools/win_port/build.ps1 -Configure                 # configure first, then build
#   pwsh -File tools/win_port/build.ps1 -Configure -Apps 0         # core only (no CLI/server)
#
# The ffmpeg and libcurl prefixes are required by the media paths; docs/windows.md explains how to
# obtain them.
[CmdletBinding()]
param(
    [switch] $Configure,
    [int] $Apps = 1,
    [int] $Tests = 0,
    [int] $Jobs = 12,
    [string] $FfmpegRoot = "D:/ffmpeg-dev/expanded/ffmpeg-master-latest-win64-gpl-shared",
    [string] $LibcurlRoot = "D:/curl-dev/expanded/curl-8.22.0_1-win64-mingw",
    [string] $SourceDir = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [string] $BuildDir = ""
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/vcvars.ps1"

if (-not $BuildDir) { $BuildDir = Join-Path $SourceDir "build-win" }
if (-not (Test-Path $FfmpegRoot)) { throw "FFmpeg prefix not found: $FfmpegRoot" }

if ($Configure) {
    $appsFlag  = if ($Apps -ne 0) { "ON" } else { "OFF" }
    $testsFlag = if ($Tests -ne 0) { "ON" } else { "OFF" }
    $arguments = @(
        "-S", $SourceDir, "-B", $BuildDir, "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CUDA_ARCHITECTURES=120a",
        "-DNINFER_BUILD_APPS=$appsFlag",
        "-DBUILD_TESTING=$testsFlag",
        "-DNINFER_FFMPEG_ROOT=$FfmpegRoot"
    )
    if (($Apps -ne 0) -or ($Tests -ne 0)) {
        if (-not (Test-Path $LibcurlRoot)) { throw "libcurl prefix not found: $LibcurlRoot" }
        $arguments += "-DNINFER_LIBCURL_ROOT=$LibcurlRoot"
    }
    Write-Host ("cmake " + ($arguments -join " "))
    & cmake @arguments
    if ($LASTEXITCODE -ne 0) { throw "configure failed with exit code $LASTEXITCODE" }
}

$log = Join-Path $BuildDir "build.log"
& cmake --build $BuildDir -j $Jobs 2>&1 | Tee-Object -FilePath $log
$status = $LASTEXITCODE
$errors = @(Select-String -Path $log -Pattern "error C[0-9]+|fatal error|error LNK" -ErrorAction SilentlyContinue)
Write-Host ("errors: " + $errors.Count)
Write-Host "BUILD_EXIT=$status"
exit $status
