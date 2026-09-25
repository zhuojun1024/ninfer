# Watchdog on/off throughput A/B. The diagnostics thread is on by default; this checks that turning
# it off changes nothing measurable, which is the claim its default rests on.
#
#   pwsh -File tools/tp_bootstrap/r88_watchdog_ab.ps1
$ErrorActionPreference = 'Stop'
Set-Location 'D:\Documents\workbench\ninfer'

$model = 'D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_final.ninfer'
$out   = 'build-win/r88'

foreach ($arm in @(@{ n = 'on'; v = '' }, @{ n = 'off'; v = '0' })) {
    if ($arm.v -ne '') { $env:NINFER_TP2_AR_WATCHDOG = $arm.v }
    else { Remove-Item Env:NINFER_TP2_AR_WATCHDOG -ErrorAction SilentlyContinue }
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    pwsh -NoProfile -File 'tools/tp_bootstrap/r62_sampling_ab.ps1' `
      -Tag ("r88-watchdog-" + $arm.n) -Model $model -DraftTokens 7 -Port 8099 -OutDir $out -Reps 6
    $code = $LASTEXITCODE
    Remove-Item Env:NINFER_TP2_AR_WATCHDOG -ErrorAction SilentlyContinue
    Write-Host ("WD_ARM {0} exit={1} elapsed_s={2}" -f $arm.n, $code, [math]::Round($sw.Elapsed.TotalSeconds, 1))

    $log   = Join-Path $out ("sampling-r88-watchdog-" + $arm.n + ".log")
    $rates = Select-String -Path $log -Pattern 'decode ([0-9.]+) tok/s' -AllMatches |
        ForEach-Object { [double]$_.Matches[0].Groups[1].Value }
    $rows = Select-String -Path $log -Pattern 'req#(\d+) done \|[^\n]*output (\d+)[^\n]*total ([0-9.]+)s' -AllMatches
    $tok = 0; $sec = 0.0
    foreach ($m in $rows.Matches) { $tok += [int]$m.Groups[2].Value; $sec += [double]$m.Groups[3].Value }
    $mean = if ($rates.Count -gt 0) { ($rates | Measure-Object -Average).Average } else { 0 }
    $agg  = if ($sec -gt 0) { $tok / $sec } else { 0 }
    Write-Host ("WD_STATS {0} n={1} mean_decode={2:N2} tok/s aggregate={3:N2} tok/s ({4} tokens / {5:N1} s)" -f `
        $arm.n, $rates.Count, $mean, $agg, $tok, $sec)
    $dumps = @(Select-String -Path $log -Pattern '\[ar-watch\]' -ErrorAction SilentlyContinue).Count
    Write-Host ("WD_DUMPS {0} ar_watch_lines={1}" -f $arm.n, $dumps)
}
Write-Host 'WATCHDOG_AB_DONE'
