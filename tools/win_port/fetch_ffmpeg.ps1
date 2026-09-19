# Fetch an MSVC-linkable FFmpeg (headers + import libraries) for the native Windows build.
# The runtime-only tree already on this machine has no include/ or lib/, so the core media decoder
# (src/media/decode) cannot be compiled against it. BtbN's shared win64 build ships both.
$ErrorActionPreference = "Stop"
$dst = "D:\ffmpeg-dev"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
$url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl-shared.zip"
$out = Join-Path $dst "ffmpeg-win64-shared.zip"
if (-not (Test-Path $out)) {
    try {
        Write-Output "direct download"
        Invoke-WebRequest -Uri $url -OutFile $out -TimeoutSec 180
    } catch {
        Write-Output "direct failed ($($_.Exception.Message)); retrying through the local proxy"
        Invoke-WebRequest -Uri $url -OutFile $out -Proxy "http://127.0.0.1:7897" -TimeoutSec 600
    }
}
Get-Item $out | Select-Object -ExpandProperty Length
$expanded = Join-Path $dst "expanded"
if (-not (Test-Path $expanded)) {
    Expand-Archive -Path $out -DestinationPath $expanded -Force
}
$root = Get-ChildItem $expanded -Directory | Select-Object -First 1
Write-Output "root=$($root.FullName)"
foreach ($p in @("include\libavcodec\avcodec.h", "include\libswscale\swscale.h",
                 "lib\avcodec.lib", "lib\avformat.lib", "lib\avutil.lib", "lib\swscale.lib")) {
    Write-Output "$([bool](Test-Path (Join-Path $root.FullName $p)))  $p"
}
