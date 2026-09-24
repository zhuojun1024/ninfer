# Sampling acceptance A/B for the section 8 draft-quantization campaign.
#
#   pwsh -File tools/tp_bootstrap/r62_sampling_ab.ps1 -Tag base -Model D:/LLM/<artifact>.ninfer -DraftTokens 5
#
# The greedy 7x160 probe of section 3.7 is content-confounded: a draft-weight change moves the
# target's tie flips, so the two arms walk different texts and acceptance moves with the text
# rather than with the proposal distribution (section 7 measured 21% vs 99.6% on one binary just by
# changing the continuation). This script keeps the serve recipe's own sampling parameters
# (temperature 0.7, top-k 20, top-p 0.8) and draws each prompt class several times, so the pooled
# acceptance averages over many independent texts and measures the draft's q itself. Only the
# ratio matters, so the metric is insensitive to machine drift. One engine at a time.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $Tag,
    [Parameter(Mandatory = $true)][string] $Model,
    [int] $DraftTokens = 5,
    [int] $Port = 8099,
    [string] $OutDir = "build-win/r62",
    [int] $Reps = 6,
    [int] $MaxTokens = 256
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$log = Join-Path $OutDir ("sampling-" + $Tag + ".log")
$jsonl = Join-Path $OutDir ("sampling-" + $Tag + "-k" + $DraftTokens + ".jsonl")

Get-Process -Name ninfer-serve -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 3

$server = Start-Process -FilePath "pwsh" -PassThru -WindowStyle Hidden -ArgumentList @(
    "-NoProfile", "-File", (Join-Path $repo "tools/win_port/serve.ps1"),
    "-Model", $Model, "-Spec", "dflash2", "-DraftTokens", "$DraftTokens",
    "-Port", "$Port", "-LogFile", $log
)
try {
    $deadline = (Get-Date).AddMinutes(8)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 5
        try {
            $health = (Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5).Content
            if ($health) { $ready = $true; break }
        } catch { }
    }
    if (-not $ready) { throw "server did not become ready; see $log" }
    Write-Host ("server ready: " + $Tag)

    $models = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/models" -TimeoutSec 30
    $modelId = $models.data[0].id

    $prompts = @(
        @{ name = "reason"; text = "A train leaves Station A at 09:15 travelling 72 km/h, and a second train leaves Station B at 09:40 travelling 96 km/h. The stations are 310 km apart. Solve this step by step, showing every calculation and checking the result." },
        @{ name = "prose";  text = "Write a detailed description of how a submarine works, about 400 words." },
        @{ name = "code";   text = "Write a Python function that parses a CSV file with quoted fields, including tests. Explain the edge cases." }
    )

    $lines = @()
    for ($rep = 1; $rep -le $Reps; $rep++) {
        foreach ($prompt in $prompts) {
            $payload = @{
                model       = $modelId
                messages    = @(@{ role = "user"; content = $prompt.text })
                max_tokens  = $MaxTokens
                temperature = 0.7
                top_k       = 20
                top_p       = 0.8
                stream      = $false
            } | ConvertTo-Json -Depth 8 -Compress
            $body = [System.Text.Encoding]::UTF8.GetBytes($payload)
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post `
                -ContentType "application/json; charset=utf-8" -Body $body -TimeoutSec 900
            $record = [ordered]@{
                rep              = $rep
                prompt           = $prompt.name
                completion_tokens = $resp.usage.completion_tokens
                draft_n          = $resp.timings.draft_n
                draft_n_accepted = $resp.timings.draft_n_accepted
            }
            $lines += ($record | ConvertTo-Json -Compress)
            Write-Host ("rep " + $rep + " " + $prompt.name + " tokens " + $resp.usage.completion_tokens +
                " draft_n " + $resp.timings.draft_n + " accepted " + $resp.timings.draft_n_accepted)
        }
    }

    Set-Content -Path $jsonl -Value $lines -Encoding UTF8
    $rows = $lines | ForEach-Object { $_ | ConvertFrom-Json }
    $drafted = ($rows | Measure-Object -Property draft_n -Sum).Sum
    $accepted = ($rows | Measure-Object -Property draft_n_accepted -Sum).Sum
    $tokens = ($rows | Measure-Object -Property completion_tokens -Sum).Sum
    $rate = if ($drafted -gt 0) { 100.0 * $accepted / $drafted } else { 0.0 }
    Write-Host ("SUMMARY tag=" + $Tag + " k=" + $DraftTokens + " runs=" + $rows.Count + " drafted=" + $drafted +
        " accepted=" + $accepted + " acceptance=" + [math]::Round($rate, 2) + "% tokens=" + $tokens)
} finally {
    Get-Process -Name ninfer-serve -ErrorAction SilentlyContinue | Stop-Process -Force
    if ($server -and -not $server.HasExited) { $server | Stop-Process -Force }
    Start-Sleep -Seconds 3
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    Write-Host ("sampling arm done: " + $Tag + " log " + $log)
}
