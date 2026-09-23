# Sweep --draft-tokens on the TP-2 DFlash2 production recipe.
# Every K starts ninfer-serve with identical production arguments, changing only --draft-tokens,
# then measures 3 prompt classes x Reps generations each.
param(
    [int[]] $DraftTokenList = @(1, 3, 5, 7, 9, 11, 13, 15),
    [int] $Reps = 2,
    [int] $MaxTokens = 256,
    [string] $Tag = '',
    [string] $Spec = 'dflash2'
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
    '--spec', 'SPEC_PLACEHOLDER', '--draft-tokens', 'K_PLACEHOLDER', '--lm-head-draft',
    '--reasoning-effort', 'medium', '--temperature', '0.7', '--top-k', '20', '--top-p', '0.8',
    '--vision', '--vision-item-tokens', '8192', '--preserve-thinking'
)

$prompts = @(
    @{ name = 'reason'; text = 'A train leaves Station A at 09:15 travelling 72 km/h, and a second train leaves Station B at 09:40 travelling 96 km/h. The stations are 310 km apart. Solve this step by step, showing every calculation and checking the result.' },
    @{ name = 'prose';  text = 'Write a detailed description of how a submarine works, about 400 words.' },
    @{ name = 'code';   text = 'Write a Python function that parses a CSV file with quoted fields, including tests. Explain the edge cases.' }
)

function Invoke-Chat([string] $prompt) {
    $body = @{ model = 'qwen3.8-27b-w4a4-draftall'; messages = @(@{ role = 'user'; content = $prompt }); max_tokens = $MaxTokens; stream = $false } | ConvertTo-Json -Depth 6
    try {
        $r = Invoke-WebRequest -Uri 'http://127.0.0.1:3456/v1/chat/completions' -Method POST -ContentType 'application/json' -Body $body -TimeoutSec 600 -UseBasicParsing -SkipHttpErrorCheck
        return $r.StatusCode
    } catch { return 'ERR' }
}

Get-Process ninfer-serve -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 3

foreach ($draftK in $DraftTokenList) {
    $log = Join-Path $outDir ('serve-K{0}{1}.log' -f $draftK, $Tag)
    Remove-Item $log, "$log.out" -ErrorAction SilentlyContinue
    $cliArgs = $baseArgs | ForEach-Object { if ($_ -eq 'K_PLACEHOLDER') { "$draftK" } elseif ($_ -eq 'SPEC_PLACEHOLDER') { $Spec } else { $_ } }
    $proc = Start-Process -FilePath $binary -ArgumentList $cliArgs -NoNewWindow -PassThru -RedirectStandardOutput "$log.out" -RedirectStandardError $log
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 2
        if ($proc.HasExited) { break }
        try { if ((Invoke-WebRequest -Uri 'http://127.0.0.1:3456/health' -TimeoutSec 3 -UseBasicParsing).StatusCode -eq 200) { $ready = $true; break } } catch { }
    }
    if (-not $ready) {
        Write-Host ('K=' + $draftK + ' NOT READY (exited=' + $proc.HasExited + '); tail:')
        Get-Content $log -Tail 4 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host ('   ' + $_) }
        if (-not $proc.HasExited) { $proc | Stop-Process -Force }
        Start-Sleep -Seconds 3
        continue
    }
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $nil = Invoke-Chat $prompts[0].text
    $decodes = @()
    $accepts = @()
    $tprs = @()
    foreach ($rep in 1..$Reps) {
        foreach ($p in $prompts) {
            $before = (Get-Content $log).Count
            $status = Invoke-Chat $p.text
            Start-Sleep -Milliseconds 300
            $line = (Get-Content $log | Select-Object -Skip $before | Select-String 'done \|' | Select-Object -Last 1).Line
            if ($line) {
                $dec = if ($line -match 'decode ([0-9.]+) tok/s') { [double]$matches[1] } else { $null }
                $acc = if ($line -match 'accepted ([\d,]+)/([\d,]+) \(([0-9.]+)%\)') { [double]$matches[3] } else { $null }
                $out = if ($line -match 'output ([\d,]+)') { $matches[1] -replace ',', '' } else { '0' }
                $drafted = if ($line -match 'accepted ([\d,]+)/([\d,]+)') { [int]($matches[2] -replace ',', '') } else { 0 }
                $rounds = if ($drafted -gt 0) { $drafted / $draftK } else { 0 }
                $tpr = if ($rounds -gt 0) { [math]::Round([double]$out / $rounds, 2) } else { 0 }
                if ($dec -ne $null) { $decodes += $dec }
                if ($acc -ne $null) { $accepts += $acc }
                if ($tpr -gt 0) { $tprs += $tpr }
                Write-Host ('K={0} {1} rep{2} HTTP {3} | decode {4} tok/s | accept {5}% | tokens/round {6} | rounds {7}' -f $draftK, $p.name, $rep, $status, $dec, $acc, $tpr, $rounds)
            } else {
                Write-Host ('K={0} {1} rep{2} HTTP {3} | no log line' -f $draftK, $p.name, $rep, $status)
            }
        }
    }
    $proc | Stop-Process -Force
    Start-Sleep -Seconds 3
    if ($decodes.Count -gt 0) {
        $md = [math]::Round(($decodes | Measure-Object -Average).Average, 1)
        $ma = [math]::Round(($accepts | Measure-Object -Average).Average, 1)
        $mt = if ($tprs.Count -gt 0) { [math]::Round(($tprs | Measure-Object -Average).Average, 2) } else { 0 }
        Write-Host ('SUMMARY K=' + $draftK + ' mean decode ' + $md + ' tok/s | mean accept ' + $ma + '% | mean tokens/round ' + $mt + ' | samples ' + $decodes.Count + ' | ' + [int]$sw.Elapsed.TotalSeconds + ' s')
    } else {
        Write-Host ('SUMMARY K=' + $draftK + ' no samples')
    }
}
Write-Host 'SWEEP_COMPLETE'
