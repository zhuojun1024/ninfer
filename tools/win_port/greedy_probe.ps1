# Greedy token-identity probe: run the same prompts with temperature 0 against one server and dump
# each completion, so two server configurations can be compared byte for byte.
#
#   pwsh -File tools/win_port/greedy_probe.ps1 -BaseUrl http://127.0.0.1:3457/v1 -OutFile out.jsonl
#
# The script never starts or stops an engine. Run exactly one engine at a time.
[CmdletBinding()]
param(
    [string] $BaseUrl = "http://127.0.0.1:3457/v1",
    [string] $Model = "",
    [int] $MaxTokens = 160,
    [int] $TimeoutSec = 900,
    [string] $OutFile = ""
)

$ErrorActionPreference = "Stop"

if (-not $Model) {
    # The OpenAI schema requires the field, and an empty string is rejected.
    $models = Invoke-RestMethod -Uri "$BaseUrl/models" -TimeoutSec 30
    $Model = $models.data[0].id
}
Write-Host ("greedy_probe base=" + $BaseUrl + " model=" + $Model + " max_tokens=" + $MaxTokens)

$prompts = @(
    "Write a concise explanation of how a paged KV cache works.",
    "Give three reasons a tensor-parallel all-reduce can be latency bound.",
    "Summarize the roofline argument for batch-one decode in two sentences.",
    "List the steps to bisect a CUDA launch failure in a multi-device kernel.",
    "Explain prefix caching and one failure mode of it.",
    "How many distinct values can a 4-bit signed integer take, and why?",
    "What is the difference between workgroup and grid synchronisation in CUDA?"
)

$lines = @()
for ($i = 0; $i -lt $prompts.Count; $i++) {
    $payload = @{
        model      = $Model
        messages   = @(@{ role = "user"; content = $prompts[$i] })
        max_tokens = $MaxTokens
        temperature = 0
        stream     = $false
    } | ConvertTo-Json -Depth 8 -Compress
    $body = [System.Text.Encoding]::UTF8.GetBytes($payload)
    $resp = Invoke-RestMethod -Uri "$BaseUrl/chat/completions" -Method Post `
        -ContentType "application/json; charset=utf-8" -Body $body -TimeoutSec $TimeoutSec
    $msg = $resp.choices[0].message
    $text = [string]$msg.content
    if ($null -ne $msg.reasoning_content) {
        $text = [string]$msg.reasoning_content + "`n<<content>>`n" + $text
    }
    $record = [ordered]@{
        index  = $i
        prompt = $prompts[$i]
        length = $text.Length
        tokens = $resp.usage.completion_tokens
        text   = $text
    }
    $lines += ($record | ConvertTo-Json -Depth 6 -Compress)
    Write-Host ("prompt " + $i + ": len " + $text.Length + " tokens " + $resp.usage.completion_tokens)
}

if ($OutFile) {
    $dir = Split-Path -Parent $OutFile
    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    Set-Content -Path $OutFile -Value $lines -Encoding UTF8
    Write-Host ("wrote " + $OutFile)
} else {
    $lines | Write-Output
}
