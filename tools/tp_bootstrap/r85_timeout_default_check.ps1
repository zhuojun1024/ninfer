# Verifies the bounded-spin default: the same injected trip must take ~8 s longer with the 10 s
# default than with an explicit NINFER_TP2_AR_TIMEOUT_MS=2000.
$ErrorActionPreference = 'Stop'
Set-Location 'D:\Documents\workbench\ninfer'
foreach ($setting in @(@{ n = 'default-10s'; e = '' }, @{ n = 'override-2s'; e = '2000' })) {
    if ($setting.e -ne '') { $env:NINFER_TP2_AR_TIMEOUT_MS = $setting.e }
    else { Remove-Item Env:NINFER_TP2_AR_TIMEOUT_MS -ErrorAction SilentlyContinue }
    $out = "build-win/r85/inject-$($setting.n)"
    $sw  = [System.Diagnostics.Stopwatch]::StartNew()
    pwsh -NoProfile -Command "& './tools/tp_bootstrap/r72_stall_injection.ps1' -Calls @(700) -OutDir '$out'" 2>&1 |
        Select-String 'INJECT call' | ForEach-Object { $_.Line }
    Write-Output ("TIMING {0} elapsed_s={1}" -f $setting.n, [math]::Round($sw.Elapsed.TotalSeconds, 1))
}
Remove-Item Env:NINFER_TP2_AR_TIMEOUT_MS -ErrorAction SilentlyContinue
Write-Output 'TIMEOUT_DEFAULT_CHECK_DONE'
