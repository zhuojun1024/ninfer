# Compare two OpenAI-compatible serving engines on identical workloads.
#
#   pwsh -File tools/win_port/bench_serve.ps1 -Label ninfer -BaseUrl http://127.0.0.1:8099/v1 -OutFile build-win/bench-ninfer.json
#   pwsh -File tools/win_port/bench_serve.ps1 -Label llama  -BaseUrl http://127.0.0.1:8080/v1 -OutFile build-win/bench-llama.json
#
# Every workload is one non-streaming chat completion and every reported number comes from the
# server's own timings block (prompt_n/prompt_ms, predicted_n/predicted_ms), so both engines are
# measured the same way. Each prompt carries a unique nonce, so prefix caching cannot fake prefill.
# The script never starts or stops an engine; run exactly one of them at a time.
[CmdletBinding()]
param(
    [string] $BaseUrl = "http://127.0.0.1:8099/v1",
    [string] $Label = "server",
    [string] $Model = "",
    [string] $OutFile = "",
    [int[]] $PrefillSizes = @(2048, 8192, 32768),
    [int] $DecodeTokens = 256,
    [int] $ContextDecodePrefill = 8192,
    [int] $ContextDecodeTokens = 128,
    [int] $TimeoutSec = 900
)

$ErrorActionPreference = "Stop"

function New-Filler {
    param([int] $ApproxTokens, [string] $Nonce)
    # The Qwen tokenizer averages roughly 1.25 tokens per word; the exact size is reported back.
    $sentence = "The quick brown fox jumps over the lazy dog while the engineer inspects memory bandwidth of the accelerator and records every counter carefully. "
    $wordsPerSentence = 23
    $sentences = [int][Math]::Ceiling(($ApproxTokens / 1.25) / $wordsPerSentence)
    return "Benchmark run $Nonce. " + ($sentence * $sentences) + " Summarize the passage in one word."
}

function New-Nonce { [guid]::NewGuid().ToString("N").Substring(0, 8) }

function Invoke-Chat {
    param([string] $Prompt, [int] $MaxTokens, [string] $Tag)
    $payload = @{
        model = $Model
        messages = @(@{ role = "user"; content = $Prompt })
        max_tokens = $MaxTokens
        stream = $false
    } | ConvertTo-Json -Depth 8 -Compress
    $body = [System.Text.Encoding]::UTF8.GetBytes($payload)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $resp = Invoke-RestMethod -Uri "$BaseUrl/chat/completions" -Method Post -ContentType "application/json; charset=utf-8" -Body $body -TimeoutSec $TimeoutSec
    $sw.Stop()
    $t = $resp.timings
    $reasoning = 0
    if ($null -ne $resp.choices[0].message.reasoning_content) { $reasoning = ([string]$resp.choices[0].message.reasoning_content).Length }
    [pscustomobject]@{
        tag                 = $Tag
        wall_s              = [Math]::Round($sw.Elapsed.TotalSeconds, 3)
        prompt_n            = $t.prompt_n
        prompt_ms           = [Math]::Round([double]$t.prompt_ms, 1)
        prompt_per_second   = [Math]::Round([double]$t.prompt_per_second, 1)
        predicted_n         = $t.predicted_n
        predicted_ms        = [Math]::Round([double]$t.predicted_ms, 1)
        predicted_per_second = [Math]::Round([double]$t.predicted_per_second, 1)
        cache_n             = $t.cache_n
        finish              = $resp.choices[0].finish_reason
        content_chars       = ([string]$resp.choices[0].message.content).Length
        reasoning_chars     = $reasoning
    }
}

if (-not $Model) {
    $models = Invoke-RestMethod -Uri "$BaseUrl/models" -TimeoutSec 30
    $Model = $models.data[0].id
}
$started = (Get-Date).ToString("s")
Write-Host "engine=$Label base=$BaseUrl model=$Model"

$results = @()
foreach ($size in $PrefillSizes) {
    $r = Invoke-Chat -Prompt (New-Filler $size (New-Nonce)) -MaxTokens 1 -Tag ("prefill_" + $size)
    $results += $r
    Write-Host ("  prefill_" + $size + "  prompt_n=" + $r.prompt_n + "  " + $r.prompt_per_second + " tok/s  (" + $r.prompt_ms + " ms)")
}

$r = Invoke-Chat -Prompt ("Benchmark run " + (New-Nonce) + ". Write one short sentence about memory bandwidth.") -MaxTokens $DecodeTokens -Tag "decode_short"
$results += $r
Write-Host ("  decode_short  predicted_n=" + $r.predicted_n + "  " + $r.predicted_per_second + " tok/s  (" + $r.predicted_ms + " ms)")

$r = Invoke-Chat -Prompt (New-Filler $ContextDecodePrefill (New-Nonce)) -MaxTokens $ContextDecodeTokens -Tag ("decode_at_" + $ContextDecodePrefill)
$results += $r
Write-Host ("  decode_at_" + $ContextDecodePrefill + "  prompt_n=" + $r.prompt_n + "  predicted_n=" + $r.predicted_n + "  " + $r.predicted_per_second + " tok/s")

$doc = [pscustomobject]@{
    engine   = $Label
    base_url = $BaseUrl
    model    = $Model
    started  = $started
    finished = (Get-Date).ToString("s")
    results  = $results
}
$json = $doc | ConvertTo-Json -Depth 8
if ($OutFile) {
    $dir = Split-Path -Parent $OutFile
    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    $json | Set-Content -Encoding UTF8 -LiteralPath $OutFile
    Write-Host "wrote $OutFile"
}
$json
