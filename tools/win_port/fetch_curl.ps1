# Fetch an MSVC-linkable libcurl (headers + import library) for the Windows product targets.
# src/product/media_acquire uses CURLOPT_PROTOCOLS_STR, so the bundle must be libcurl >= 7.85.
$ErrorActionPreference = "Stop"
$dst = "D:\curl-dev"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
$page = Join-Path $dst "index.html"
function Get-Via($url, $out) {
    try { Invoke-WebRequest -Uri $url -OutFile $out -TimeoutSec 120 }
    catch {
        Write-Output "direct failed ($($_.Exception.Message)); proxy retry"
        Invoke-WebRequest -Uri $url -OutFile $out -Proxy "http://127.0.0.1:7897" -TimeoutSec 600
    }
}
if (-not (Test-Path $page)) { Get-Via "https://curl.se/windows/" $page }
$html = Get-Content $page -Raw
$href = [regex]::Match($html, 'href="([^"]*win64-mingw\.zip)"').Groups[1].Value
if (-not $href) { throw "no win64-mingw.zip link found on the curl.se/windows page" }
if ($href -notmatch '^https?://') { $href = "https://curl.se/windows/" + $href.TrimStart('.', '/') }
Write-Output "url=$href"
$zip = Join-Path $dst ([System.IO.Path]::GetFileName($href))
if (-not (Test-Path $zip)) { Get-Via $href $zip }
Get-Item $zip | Select-Object -ExpandProperty Length
$expanded = Join-Path $dst "expanded"
if (-not (Test-Path $expanded)) { Expand-Archive -Path $zip -DestinationPath $expanded -Force }
$root = Get-ChildItem $expanded -Directory | Select-Object -First 1
Write-Output "root=$($root.FullName)"
Get-ChildItem $root.FullName -Recurse -Include "curl.h","*.lib","*.dll.a","libcurl.dll" |
    ForEach-Object { $_.FullName.Replace($root.FullName, "") }
