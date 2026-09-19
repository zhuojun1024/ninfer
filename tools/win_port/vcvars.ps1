# Import the MSVC x64 build environment into the current PowerShell session, which is what lets the
# other win_port scripts call cmake, ninja and nvcc directly instead of wrapping cmd.
# Usage:  . "$PSScriptRoot/vcvars.ps1"

$ErrorActionPreference = "Stop"

$vcvars = "D:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }

$capture = 'call "' + $vcvars + '" >nul 2>&1 && set'
cmd /c $capture | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path ("Env:" + $matches[1]) -Value $matches[2] }
}
if (-not $env:VCToolsInstallDir) { throw "importing the vcvars64 environment failed" }
