# Prefill rate curve for DFlash2 K=7 as the context deepens from 0 to the configured maximum.
# One process, one growing document: request i sends segments 1..i, so the shared prefix is reused
# and the reported prefill rate covers only the newly added tokens - the rate at that depth.
param(
    [int] $Steps = 16,
    [int] $TargetTokensPerStep = 15000,
    [string] $Spec = 'dflash2',
    [int] $DraftTokens = 7
)
$ErrorActionPreference = 'Stop'
$root   = 'D:\Documents\workbench\ninfer'
$binary = 'C:\ninfer\ninfer-serve.exe'
$outDir = Join-Path $root 'build-win\_ksweep'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$env:PATH = 'D:/ffmpeg-dev/expanded/ffmpeg-master-latest-win64-gpl-shared/bin;D:/curl-dev/expanded/curl-8.22.0_1-win64-mingw/bin;' + $env:PATH

$cliArgs = @(
    'D:/LLM/qwen3_8_27b_w4a4_dflash2_draftall.ninfer',
    '--devices', '0,1', '--max-context', '245760', '--port', '3456', '--host', '0.0.0.0',
    '--kv-dtype', 'k8v4', '--log-level', 'info', '--max-concurrency', '1', '--prefill-chunk', '1024',
    '--max-pending-requests', '16', '--host-state-slots', '32', '--host-kv-mib', '24576',
    '--max-private-continuations', '4', '--chat-template', 'D:/LLM/chat_template.jinja',
    '--spec', $Spec, '--draft-tokens', "$DraftTokens", '--lm-head-draft',
    '--reasoning-effort', 'medium', '--temperature', '0.7', '--top-k', '20', '--top-p', '0.8',
    '--vision', '--vision-item-tokens', '8192', '--preserve-thinking'
)

$base = 'Remote sensing satellites downlink imagery in scheduled passes, and ground stations archive every scene for later analysis. '
$repeats = [math]::Max(1, [int]([math]::Round($TargetTokensPerStep * 0.75 / 16)))
$segments = @()
for ($i = 1; $i -le $Steps; $i++) {
    $segments += ('Segment ' + $i + '. ' + ($base * $repeats))
}

function Invoke-Prefill([string] $text) {
    $body = @{ model = 'qwen3.8-27b-w4a4-draftall'; messages = @(@{ role = 'user'; content = $text }); max_tokens = 1; stream = $false } | ConvertTo-Json -Depth 6
    try {
        $r = Invoke-WebRequest -Uri 'http://127.0.0.1:3456/v1/chat/completions' -Method POST -ContentType 'application/json' -Body $body -TimeoutSec 1800 -UseBasicParsing -SkipHttpErrorCheck
        return $r.StatusCode
    } catch { return 'ERR' }
}

Get-Process ninfer-serve -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 3
$log = Join-Path $outDir ('curve-{0}-K{1}.log' -f $Spec, $DraftTokens)
Remove-Item $log, "$log.out" -ErrorAction SilentlyContinue
$proc = Start-Process -FilePath $binary -ArgumentList $cliArgs -NoNewWindow -PassThru -RedirectStandardOutput "$log.out" -RedirectStandardError $log
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 2
    if ($proc.HasExited) { break }
    try { if ((Invoke-WebRequest -Uri 'http://127.0.0.1:3456/health' -TimeoutSec 3 -UseBasicParsing).StatusCode -eq 200) { $ready = $true; break } } catch { }
}
if (-not $ready) { Write-Host 'NOT READY'; Get-Content $log -Tail 3 | ForEach-Object { Write-Host $_ }; exit 1 }
Write-Host 'server ready; starting curve'

$doc = ''
$total = [System.Diagnostics.Stopwatch]::StartNew()
for ($step = 1; $step -le $Steps; $step++) {
    $doc = $doc + $segments[$step - 1]
    $before = (Get-Content $log).Count
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $status = Invoke-Prefill $doc
    Start-Sleep -Milliseconds 400
    $line = (Get-Content $log | Select-Object -Skip $before | Select-String -Pattern 'done \|' | Select-Object -Last 1).Line
    $secs = [math]::Round($sw.Elapsed.TotalSeconds, 1)
    if ($line) {
        $ptok = if ($line -match 'prompt ([\d,]+)') { [int]($matches[1] -replace ',','') } else { 0 }
        $cach = if ($line -match 'cache ([\d,]+) \(') { [int]($matches[1] -replace ',','') } else { 0 }
        $new  = if ($line -match 'prefill [0-9.]+k? tok/s \(([\d,]+) tok\)') { [int]($matches[1] -replace ',','') } else { 0 }
        $pf   = if ($line -match 'prefill ([0-9.]+)(k?) tok/s') { if ($matches[2] -eq 'k') { [double]$matches[1] * 1000 } else { [double]$matches[1] } } else { 0 }
        Write-Host ('step {0,2} | prompt {1,7} | from cache {2,7} | prefilled {3,6} | depth {4,7}-{5,7} | {6,8} tok/s | {7,5} s | HTTP {8}' -f $step, $ptok, $cach, $new, ($ptok - $new), $ptok, [math]::Round($pf, 0), $secs, $status)
    } else {
        Write-Host ('step {0,2} | HTTP {1} | no log line | {2} s' -f $step, $status, $secs)
    }
}
Write-Host ('curve total ' + [int]$total.Elapsed.TotalSeconds + ' s')
$proc | Stop-Process -Force
Write-Host 'CURVE_COMPLETE'
