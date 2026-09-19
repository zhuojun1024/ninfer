# Build and run the TP-2 mapped-pinned spike (tools/win_port/tp_mapped_spike.cu).
#   pwsh -File tools/win_port/spike.ps1
[CmdletBinding()]
param(
    [int] $DeviceA = 0,
    [int] $DeviceB = 1,
    [string] $SourceDir = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/vcvars.ps1"

$build = Join-Path $SourceDir "build-win"
New-Item -ItemType Directory -Force -Path $build | Out-Null
$exe = Join-Path $build "tp_mapped_spike.exe"
& nvcc -arch=sm_120a -O2 -Xcompiler=/utf-8 -o $exe (Join-Path $PSScriptRoot "tp_mapped_spike.cu")
if ($LASTEXITCODE -ne 0) { throw "nvcc failed with exit code $LASTEXITCODE" }
& $exe $DeviceA $DeviceB
exit $LASTEXITCODE
