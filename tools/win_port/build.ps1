# Configure and build NInfer natively on Windows.
#
#   pwsh -File tools/win_port/build.ps1                            # incremental build
#   pwsh -File tools/win_port/build.ps1 -Configure                 # configure first, then build
#   pwsh -File tools/win_port/build.ps1 -Configure -Apps 0         # core only (no CLI/server)
#   pwsh -File tools/win_port/build.ps1 -SkipProbe                 # skip the pipe preflight
#
# The ffmpeg and libcurl prefixes are required by the media paths; docs/windows.md explains how to
# obtain them.
#
# Two properties this script has to have on Windows, both learned the hard way:
#
# 1. Nothing pipes a native tool into a PowerShell cmdlet, and every child writes to a log file
#    through Start-Process redirection instead of a pipeline. `& cmake ... | Tee-Object` leaves
#    cmake/ninja attached to this shell's console pipe; a caller that does not drain that pipe
#    (an agent harness running the script as a background job) then blocks the build once the
#    buffer fills.
# 2. The build is bounded, and it preflights the one environment failure that cannot report
#    itself. ninja captures every child's stdout through a pipe it creates itself, so a sandbox
#    that denies pipe creation makes ninja block *before* it starts the first job: no output, no
#    compiler process, zero CPU, forever. The preflight reproduces that with a one-edge synthetic
#    project and fails in seconds with an actionable message instead of hanging for a full timeout.
[CmdletBinding()]
param(
    [switch] $Configure,
    [int] $Apps = 1,
    [int] $Tests = 0,
    [int] $Jobs = 12,
    [int] $TimeoutSec = 2700,
    [switch] $SkipProbe,
    [string] $FfmpegRoot = "D:/ffmpeg-dev/expanded/ffmpeg-master-latest-win64-gpl-shared",
    [string] $LibcurlRoot = "D:/curl-dev/expanded/curl-8.22.0_1-win64-mingw",
    [string] $SourceDir = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [string] $BuildDir = ""
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/vcvars.ps1"

if (-not $BuildDir) { $BuildDir = Join-Path $SourceDir "build-win" }
if (-not (Test-Path $FfmpegRoot)) { throw "FFmpeg prefix not found: $FfmpegRoot" }
$log = Join-Path $BuildDir "build.log"
$errLog = Join-Path $BuildDir "build.err.log"

# Quotes an argument list the way Start-Process needs it: it joins the array with spaces and does
# not quote elements, so any element with a space would otherwise split into two arguments.
function Format-ArgumentList([string[]] $Arguments) {
    return (($Arguments | ForEach-Object { if ($_ -match "\s") { '"' + $_ + '"' } else { $_ } }) -join " ")
}

# Runs one native tool with its stdout/stderr bound to files and a hard wall-clock limit. Returns
# the exit code; a timeout kills the process and returns 124.
function Invoke-Bounded([string] $Exe, [string[]] $Arguments, [string] $LogFile, [string] $What) {
    $started = Get-Date
    Write-Host ("{0}: {1} -> {2}" -f $What, $Exe, $LogFile)
    $process = Start-Process -FilePath $Exe -ArgumentList (Format-ArgumentList $Arguments) `
        -NoNewWindow -PassThru -RedirectStandardOutput $LogFile -RedirectStandardError "$LogFile.err"
    if (-not $process.WaitForExit($TimeoutSec * 1000)) {
        $process.Kill()
        $process.WaitForExit()
        Write-Host ("{0} exceeded {1} s and was killed; partial output: {2}" -f $What, $TimeoutSec, $LogFile)
        exit 124
    }
    $seconds = [int]((Get-Date) - $started).TotalSeconds
    $status = $process.ExitCode
    Write-Host ("{0}: exit {1} after {2} s" -f $What, $status, $seconds)
    if ($status -ne 0) {
        $errors = @(Select-String -Path $LogFile, "$LogFile.err" `
            -Pattern "error C[0-9]+|fatal error|error LNK|FAILED:" -ErrorAction SilentlyContinue)
        foreach ($error in ($errors | Select-Object -Last 40)) { Write-Host $error.Line }
        Write-Host ("full log: {0} (and {0}.err)" -f $LogFile)
        exit $status
    }
}

# Reproduces the sandbox failure described above with a one-edge project. Returns $true when ninja
# can start a child at all.
function Test-ChildProcessPipes {
    $probe = Join-Path $BuildDir "_pipe_probe"
    $null = New-Item -ItemType Directory -Force -Path $probe
    Set-Content -Path (Join-Path $probe "build.ninja") -Encoding ascii @'
rule noop
  command = cmd /c echo ok > $out
build probe.txt: noop
'@
    Remove-Item (Join-Path $probe "probe.txt") -ErrorAction SilentlyContinue
    $OUT = Join-Path $probe "probe.out"
    $ERR = Join-Path $probe "probe.err"
    $process = Start-Process -FilePath "ninja" -ArgumentList "-C $probe" -NoNewWindow -PassThru `
        -RedirectStandardOutput $OUT -RedirectStandardError $ERR
    if (-not $process.WaitForExit(8000)) {
        $process.Kill()
        $process.WaitForExit()
        return $false
    }
    return ($process.ExitCode -eq 0)
}

if (-not $SkipProbe -and -not (Test-ChildProcessPipes)) {
    Write-Host ""
    Write-Host "build: ninja cannot start a child process here."
    Write-Host "  ninja gives every child a pipe for its stdout, and this sandbox denies pipe creation,"
    Write-Host "  so ninja blocks before the first job: no output, no compiler process, zero CPU."
    Write-Host "  Run this build from a normal terminal, or re-run it with the sandbox disabled."
    Write-Host "  (-SkipProbe bypasses this check.)"
    exit 3
}

# A running server holds apps/ninfer-serve.exe open, and the link step only fails with LNK1104 after
# the whole build has already run. Refuse up front instead.
$servers = @(Get-Process ninfer-serve -ErrorAction SilentlyContinue)
if ($servers.Count -ne 0) {
    $pids = ($servers | ForEach-Object { $_.Id }) -join ", "
    Write-Host ("build: ninfer-serve is running (pid {0})" -f $pids)
    Write-Host "  it holds apps/ninfer-serve.exe open, so the link step would fail with LNK1104."
    Write-Host "  stop it first: pwsh -File tools/win_port/serve.ps1 -Stop"
    exit 4
}

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
    Invoke-Bounded "cmake" $arguments $log "configure"
}

Invoke-Bounded "cmake" @("--build", $BuildDir, "-j", "$Jobs") $log "build"

$errors = @(Select-String -Path $log, "$log.err" -Pattern "error C[0-9]+|fatal error|error LNK|FAILED:" `
    -ErrorAction SilentlyContinue)
if ($errors.Count -ne 0) {
    Write-Host ("build ok with {0} diagnostic line(s); see {1}" -f $errors.Count, $log)
}
Write-Host "BUILD_EXIT=0"
exit 0
