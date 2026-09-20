# Measure the prefill and decode context curves of a running ninfer-serve.
#
#   pwsh -File tools/win_port/context_curve.ps1 -BaseUrl http://127.0.0.1:3457/v1 -Label tp2-mtp0
#        -OutFile profiles/bench/tp2_context_curve/mtp0.json -Sizes '128,...,196608'
#
# One request per (size, repetition) yields both numbers from the response timings block, which is
# the published single-request methodology:
#   prefill phase = prompt_n / prompt_ms          (prompt_n excludes any cached prefix)
#   decode phase  = (predicted_n - 1) / predicted_ms
# prompt_ms is the prefill wall span (prepare/tokenize is reported separately and excluded) and
# predicted_ms is the decode-round wall span, matching docs/performance/methodology.md.
# Every request carries a unique nonce in both the system and user messages, so prefix reuse cannot
# fake a prefill: a root-only cache can still match the leading template tokens, which is why the
# true context is read from usage.prompt_tokens and reported separately from prompt_n.
# The script never starts or stops an engine.
[CmdletBinding()]
param(
    [string] $BaseUrl = "http://127.0.0.1:3457/v1",
    [string] $Model = "",
    [string] $Sizes = "128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,196608",
    [int]    $DecodeTokens = 256,
    [int]    $Repetitions = 3,
    [int]    $TimeoutSec = 2400,
    [string] $Label = "server",
    [string] $OutFile = "",
    [switch] $SkipWarmup
)

$ErrorActionPreference = "Stop"

$sizeList = @($Sizes -split "[,\s]+" | Where-Object { $_ -ne "" } | ForEach-Object { [int] $_ })
if ($sizeList.Count -eq 0) { throw "-Sizes is empty" }

# The Qwen tokenizer averages roughly 1.25 tokens per word; the preflight probes below turn that
# estimate into an exact tokens-per-sentence slope for this tokenizer and chat template.
$sentence    = "The quick brown fox jumps over the lazy dog while the engineer inspects memory bandwidth of the accelerator and records every counter carefully. "
$instruction = " Analyze the passage above in full detail: describe its structure, its implications for memory-bandwidth-bound inference, and every engineering consequence you can derive. Do not stop before you have covered every detail."
$counter     = 0

function New-Nonce { param([int] $Value) return ("#{0:D8}" -f $Value) }

function New-Prompt {
    param([int] $Sentences, [string] $Nonce)
    $body = ""
    if ($Sentences -gt 0) { $body = $sentence * $Sentences }
    return "Benchmark run $Nonce. " + $body + $instruction
}

function Invoke-Chat {
    param([string] $Prompt, [int] $MaxTokens, [string] $Tag, [int] $Target, [string] $Nonce)
    $payload = @{
        model      = $Model
        messages   = @(
            @{ role = "system"; content = "Benchmark session $Nonce." },
            @{ role = "user"; content = $Prompt }
        )
        max_tokens = $MaxTokens
        stream     = $false
    } | ConvertTo-Json -Depth 8 -Compress
    $body = [System.Text.Encoding]::UTF8.GetBytes($payload)
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $response = Invoke-RestMethod -Uri "$BaseUrl/chat/completions" -Method Post -ContentType "application/json; charset=utf-8" -Body $body -TimeoutSec $TimeoutSec
    $watch.Stop()
    $t = $response.timings
    if ($null -eq $t) { throw "response for $Tag has no timings block" }
    $draftN = 0; $draftAccepted = 0
    if ($null -ne $t.draft_n) { $draftN = [int] $t.draft_n }
    if ($null -ne $t.draft_n_accepted) { $draftAccepted = [int] $t.draft_n_accepted }
    $total = 0
    if ($null -ne $response.usage) { $total = [int] $response.usage.prompt_tokens }
    if ($total -le 0) { $total = [int] $t.prompt_n + [int] $t.cache_n }
    return [pscustomobject]@{
        target               = $Target
        tag                  = $Tag
        wall_s               = [Math]::Round($watch.Elapsed.TotalSeconds, 3)
        prompt_total         = $total
        prompt_n             = [int] $t.prompt_n
        cache_n              = [int] $t.cache_n
        prompt_ms            = [Math]::Round([double] $t.prompt_ms, 3)
        prompt_per_second    = [Math]::Round([double] $t.prompt_per_second, 2)
        predicted_n          = [int] $t.predicted_n
        predicted_ms         = [Math]::Round([double] $t.predicted_ms, 3)
        predicted_per_second = [Math]::Round([double] $t.predicted_per_second, 2)
        draft_n              = $draftN
        draft_n_accepted     = $draftAccepted
        finish_reason        = [string] $response.choices[0].finish_reason
        error                = $null
    }
}

function Get-Stats {
    param([double[]] $Values)
    if ($null -eq $Values -or $Values.Count -eq 0) { return [pscustomobject]@{ n = 0; mean = 0.0; sd = 0.0 } }
    $mean = ($Values | Measure-Object -Average).Average
    $sd = 0.0
    if ($Values.Count -gt 1) {
        $sum = 0.0
        foreach ($value in $Values) { $sum += ($value - $mean) * ($value - $mean) }
        $sd = [Math]::Sqrt($sum / ($Values.Count - 1))
    }
    return [pscustomobject]@{ n = $Values.Count; mean = $mean; sd = $sd }
}

if (-not $Model) {
    $models = Invoke-RestMethod -Uri "$BaseUrl/models" -TimeoutSec 30
    $Model = $models.data[0].id
}
$startedAt = (Get-Date).ToString("s")
Write-Host "context_curve label=$Label base=$BaseUrl model=$Model sizes=$($sizeList -join ",") decode_tokens=$DecodeTokens reps=$Repetitions"

# --- calibration: two cheap prefill probes give prompt_n = overhead + slope * sentences ---------
$probeA = 48
$probeB = 384
$calA = Invoke-Chat -Prompt (New-Prompt $probeA (New-Nonce $counter)) -MaxTokens 1 -Tag "calib_a" -Target 0 -Nonce (New-Nonce $counter)
$counter++
$calB = Invoke-Chat -Prompt (New-Prompt $probeB (New-Nonce $counter)) -MaxTokens 1 -Tag "calib_b" -Target 0 -Nonce (New-Nonce $counter)
$counter++
$slope    = (($calB.prompt_total) - ($calA.prompt_total)) / [double] ($probeB - $probeA)
$overhead = $calA.prompt_total - $slope * $probeA
Write-Host ("calibration: sentence={0:N3} tok, overhead={1:N1} tok (probes {2}/{3})" -f $slope, $overhead, $calA.prompt_total, $calB.prompt_total)

if ($OutFile) {
    $dir = Split-Path -Parent $OutFile
    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
}
$jsonl = if ($OutFile) { [System.IO.Path]::ChangeExtension($OutFile, ".jsonl") } else { "" }
if ($jsonl -and (Test-Path $jsonl)) { Remove-Item -LiteralPath $jsonl -Force }

if (-not $SkipWarmup) {
    # One ordinary request primes the decode graph so capture cost does not land in the first point.
    $warm = Invoke-Chat -Prompt (New-Prompt 16 (New-Nonce $counter)) -MaxTokens 32 -Tag "warmup" -Target 0 -Nonce (New-Nonce $counter)
    $counter++
    Write-Host ("warmup: prompt_total={0} prompt_n={1} cache_n={2} decode={3} tok/s" -f $warm.prompt_total, $warm.prompt_n, $warm.cache_n, $warm.predicted_per_second)
}

# --- sweep ---------------------------------------------------------------------------------------
$records = New-Object System.Collections.Generic.List[object]
foreach ($target in $sizeList) {
    $tolerance = [Math]::Max(4.0, 0.005 * $target)
    $sentences = [int] [Math]::Round(($target - $overhead) / $slope)
    if ($sentences -lt 0) { $sentences = 0 }

    if ($target -le 8192) {
        # Cheap sizes: refine the sentence count against the real tokenizer before measuring.
        for ($attempt = 0; $attempt -lt 4; $attempt++) {
            $probe = Invoke-Chat -Prompt (New-Prompt $sentences (New-Nonce $counter)) -MaxTokens 1 -Tag "refine_$target" -Target 0 -Nonce (New-Nonce $counter)
            $counter++
            $delta = [Math]::Abs($probe.prompt_total - $target)
            Write-Host ("  refine target=$target sentences=$sentences prompt_total=$($probe.prompt_total) delta=$([Math]::Round($delta,1))")
            if ($delta -le $tolerance -or $sentences -eq 0) { break }
            $step = [int] [Math]::Round(($target - $probe.prompt_total) / $slope)
            if ($step -eq 0) { break }
            $sentences = [Math]::Max(0, $sentences + $step)
        }
    }

    for ($rep = 1; $rep -le $Repetitions; $rep++) {
        $tag = "size_" + $target + "_rep$rep"
        try {
            $record = Invoke-Chat -Prompt (New-Prompt $sentences (New-Nonce $counter)) -MaxTokens $DecodeTokens -Tag $tag -Target $target -Nonce (New-Nonce $counter)
        } catch {
            $record = [pscustomobject]@{
                target = $target; tag = $tag; wall_s = 0.0; prompt_total = 0; prompt_n = 0; cache_n = 0; prompt_ms = 0.0;
                prompt_per_second = 0.0; predicted_n = 0; predicted_ms = 0.0; predicted_per_second = 0.0;
                draft_n = 0; draft_n_accepted = 0;
                finish_reason = ""; error = $_.Exception.Message
            }
        }
        $counter++
        $records.Add($record)
        if ($OutFile) { ($record | ConvertTo-Json -Depth 4 -Compress) | Add-Content -LiteralPath $jsonl -Encoding utf8 }
        if ($record.error) {
            Write-Host ("  target=$target rep=$rep FAILED: $($record.error)")
        } else {
            Write-Host ("  target=$target rep=$rep context=$($record.prompt_total) computed=$($record.prompt_n) cache=$($record.cache_n) prefill=$($record.prompt_per_second) tok/s decode=$($record.predicted_per_second) tok/s (n=$($record.predicted_n), $($record.finish_reason))")
        }
    }
}

# --- summary -------------------------------------------------------------------------------------
$summary = New-Object System.Collections.Generic.List[object]
foreach ($target in $sizeList) {
    $rows = @($records | Where-Object { $_.target -eq $target -and -not $_.error })
    if ($rows.Count -eq 0) { continue }
    $prefillStats = Get-Stats @($rows | ForEach-Object { [double] $_.prompt_per_second })
    $totalStats   = Get-Stats @($rows | ForEach-Object { [double] $_.prompt_total })
    $cacheStats   = Get-Stats @($rows | ForEach-Object { [double] $_.cache_n })
    $decodeRows   = @($rows | Where-Object { [int] $_.predicted_n -ge ($DecodeTokens - 2) })
    $decodeStats  = Get-Stats @($decodeRows | ForEach-Object { [double] $_.predicted_per_second })
    $drafted      = ($rows | Measure-Object -Property draft_n -Sum).Sum
    $accepted     = ($rows | Measure-Object -Property draft_n_accepted -Sum).Sum
    $acceptance   = if ($drafted -gt 0) { [Math]::Round(100.0 * $accepted / $drafted, 1) } else { 0.0 }
    $summary.Add([pscustomobject]@{
        target        = $target
        context       = [Math]::Round($totalStats.mean, 1)
        cache_n       = [Math]::Round($cacheStats.mean, 1)
        prefill_tps   = [Math]::Round($prefillStats.mean, 1)
        prefill_sd    = [Math]::Round($prefillStats.sd, 1)
        decode_tps    = [Math]::Round($decodeStats.mean, 2)
        decode_sd     = [Math]::Round($decodeStats.sd, 2)
        decode_reps   = $decodeStats.n
        acceptance    = $acceptance
    })
}

$doc = [pscustomobject]@{
    label         = $Label
    base_url      = $BaseUrl
    model         = $Model
    started       = $startedAt
    finished      = (Get-Date).ToString("s")
    decode_tokens = $DecodeTokens
    repetitions   = $Repetitions
    calibration   = [pscustomobject]@{ tokens_per_sentence = [Math]::Round($slope, 4); overhead_tokens = [Math]::Round($overhead, 2) }
    summary       = $summary
    records       = $records
}
if ($OutFile) {
    ($doc | ConvertTo-Json -Depth 6) | Set-Content -LiteralPath $OutFile -Encoding utf8
    Write-Host "wrote $OutFile"
}

Write-Host ""
Write-Host "| context tokens | cached | prefill tok/s | sd | decode tok/s | sd | decode reps | accept % |"
Write-Host "|---:|---:|---:|---:|---:|---:|---:|---:|"
foreach ($row in $summary) {
    Write-Host ("| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} |" -f $row.context, $row.cache_n, $row.prefill_tps, $row.prefill_sd, $row.decode_tps, $row.decode_sd, $row.decode_reps, $row.acceptance)
}
