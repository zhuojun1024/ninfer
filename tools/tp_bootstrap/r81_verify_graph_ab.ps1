# B7 acceptance for the DFlash2 verify CUDA Graph (PLAN section 3.11 item 2).
#
#   pwsh -File tools/tp_bootstrap/r81_verify_graph_ab.ps1
#
# NINFER_TP2_VERIFY_GRAPH=0 keeps the same window and the same envelope but launches it eagerly, so the
# two arms differ only in the launch. Equal-output throughput is read from the serve logs' per-request
# decode rate; acceptance must match because the mathematics is identical.
$ErrorActionPreference = 'Stop'
Set-Location 'D:\Documents\workbench\ninfer'

$model = 'D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_final.ninfer'
$out   = 'build-win/r81'

foreach ($arm in @(@{ n = 'graph'; v = '1' }, @{ n = 'eager'; v = '0' })) {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $env:NINFER_TP2_VERIFY_GRAPH = $arm.v
    pwsh -NoProfile -File 'tools/tp_bootstrap/r62_sampling_ab.ps1' `
      -Tag ("r81-verify-" + $arm.n) -Model $model -DraftTokens 7 -Port 8099 -OutDir $out -Reps 6
    $code = $LASTEXITCODE
    Remove-Item Env:NINFER_TP2_VERIFY_GRAPH -ErrorAction SilentlyContinue
    Write-Host ("VERIFY_ARM {0} exit={1} elapsed_s={2}" -f $arm.n, $code, [math]::Round($sw.Elapsed.TotalSeconds, 1))

    $log = Join-Path $out ("sampling-r81-verify-" + $arm.n + ".log")
    $rates = Select-String -Path $log -Pattern 'decode ([0-9.]+) tok/s' -AllMatches |
        ForEach-Object { [double]$_.Matches[0].Groups[1].Value }
    if ($rates.Count -gt 0) {
        $mean = ($rates | Measure-Object -Average).Average
        Write-Host ("VERIFY_RATE {0} n={1} mean={2:N2} tok/s min={3:N2} max={4:N2}" -f `
            $arm.n, $rates.Count, $mean, ($rates | Measure-Object -Minimum).Minimum,
            ($rates | Measure-Object -Maximum).Maximum)
    } else {
        Write-Host ("VERIFY_RATE {0} no decode lines in {1}" -f $arm.n, $log)
    }
}
Write-Host 'VERIFY_GRAPH_AB_DONE'
