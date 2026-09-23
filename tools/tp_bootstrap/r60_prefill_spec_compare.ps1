# Compare prefill throughput across speculative backends on the TP-2 production recipe.
# Identical production arguments in all three configs; only the spec flags differ. Prefix reuse is
# disabled so every request prefills the full prompt, and each config prefills the same text.
param(
    [string[]] $Configs = @('dflash2', 'mtp', 'plain'),
    [int] $Cycles = 2,
    [int[]] $PromptTokens = @(2048, 8192),
    [string] $Tag = ''
)
$ErrorActionPreference = 'Stop'
$root   = 'D:\Documents\workbench\ninfer'
$binary = 'C:\ninfer\ninfer-serve.exe'
$outDir = Join-Path $root 'build-win\_ksweep'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$env:PATH = 'D:/ffmpeg-dev/expanded/ffmpeg-master-latest-win64-gpl-shared/bin;D:/curl-dev/expanded/curl-8.22.0_1-win64-mingw/bin;' + $env:PATH

$baseArgs = @(
    'D:/LLM/qwen3_8_27b_w4a4_dflash2_draftall.ninfer',
    '--devices', '0,1', '--max-context', '245760', '--port', '3456', '--host', '0.0.0.0',
    '--kv-dtype', 'k8v4', '--log-level', 'info', '--max-concurrency', '1', '--prefill-chunk', '1024',
    '--max-pending-requests', '16', '--host-state-slots', '32', '--host-kv-mib', '24576',
    '--max-private-continuations', '4', '--chat-template', 'D:/LLM/chat_template.jinja',
    '--reasoning-effort', 'medium', '--temperature', '0.7', '--top-k', '20', '--top-p', '0.8',
    '--vision', '--vision-item-tokens', '8192', '--preserve-thinking'
)
$specArgs = @{
    'dflash2' = @('--spec', 'dflash2', '--draft-tokens', '7', '--lm-head-draft')
    'mtp'     = @('--spec', 'mtp', '--draft-tokens', '3', '--lm-head-draft')
    'plain'   = @()
}

$sentences = @{
    2048 = 'The submarine hull is divided into watertight compartments so that flooding in one section does not sink the whole boat. '
    8192 = 'Coral reefs grow slowly as polyps deposit calcium carbonate skeletons, building structures that shelter a quarter of all marine species. '
}
function New-Prompt([int] $target) {
    $sentence = $sentences[$target]
    $wordsPerRepeat = 22
    $repeats = [math]::Max(1, [int]([math]::Round($target * 0.75 / $wordsPerRepeat)))
    $body = $sentence * $repeats
    return ('Summarize the following text in one sentence.' + [char]10 + [char]10 + $body)
}

function Invoke-Prefill([string] $prompt) {
    $body = @{ model = 'qwen3.8-27b-w4a4-draftall'; messages = @(@{ role = 'user'; content = $prompt }); max_tokens = 16; stream = $false } | ConvertTo-Json -Depth 6
    try {
        $r = Invoke-WebRequest -Uri 'http://127.0.0.1:3456/v1/chat/completions' -Method POST -ContentType 'application/json' -Body $body -TimeoutSec 900 -UseBasicParsing -SkipHttpErrorCheck
        return $r.StatusCode
    } catch { return 'ERR' }
}

Get-Process ninfer-serve -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 3

foreach ($cycle in 1..$Cycles) {
    foreach ($cfg in $Configs) {
        $log = Join-Path $outDir ('prefill-{0}-c{1}{2}.log' -f $cfg, $cycle, $Tag)
        Remove-Item $log, "$log.out" -ErrorAction SilentlyContinue
        $cliArgs = $baseArgs + $specArgs[$cfg]
        $proc = Start-Process -FilePath $binary -ArgumentList $cliArgs -NoNewWindow -PassThru -RedirectStandardOutput "$log.out" -RedirectStandardError $log
        $ready = $false
        for ($i = 0; $i -lt 60; $i++) {
            Start-Sleep -Seconds 2
            if ($proc.HasExited) { break }
            try { if ((Invoke-WebRequest -Uri 'http://127.0.0.1:3456/health' -TimeoutSec 3 -UseBasicParsing).StatusCode -eq 200) { $ready = $true; break } } catch { }
        }
        if (-not $ready) {
            Write-Host ('cycle' + $cycle + ' ' + $cfg + ' NOT READY'); Get-Content $log -Tail 3 | ForEach-Object { Write-Host ('   ' + $_) }
            if (-not $proc.HasExited) { $proc | Stop-Process -Force }
            Start-Sleep -Seconds 3
            continue
        }
        foreach ($target in $PromptTokens) {
            $prompt = New-Prompt $target
            $before = (Get-Content $log).Count
            $status = Invoke-Prefill $prompt
            Start-Sleep -Milliseconds 300
            $line = (Get-Content $log | Select-Object -Skip $before | Select-String -Pattern 'done \|' | Select-Object -Last 1).Line
            if ($line) {
                $ptok = if ($line -match 'prompt ([\d,]+)') { $matches[1] -replace ',', '' } else { '?' }
                $pfr  = if ($line -match 'prefill ([0-9.]+) tok/s') { $matches[1] } else { '?' }
                $ttft = if ($line -match 'TTFT ([0-9.]+) ms') { $matches[1] } else { '?' }
                Write-Host ('c{0} {1,-7} target {2,5} | HTTP {3} | prompt {4} tok | prefill {5} tok/s | TTFT {6} ms' -f $cycle, $cfg, $target, $status, $ptok, $pfr, $ttft)
            } else {
                Write-Host ('c{0} {1,-7} target {2,5} | HTTP {3} | no log line' -f $cycle, $cfg, $target, $status)
            }
        }
        $proc | Stop-Process -Force
        Start-Sleep -Seconds 3
    }
}
Write-Host 'PREFILL_COMPLETE'
