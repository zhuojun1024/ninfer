# Verifies the watchdog default: the same injected trip must leave exactly one detailed dump with
# the thread on by default, and none at all with NINFER_TP2_AR_WATCHDOG=0.
$ErrorActionPreference = 'Stop'
Set-Location 'D:\Documents\workbench\ninfer'
foreach ($setting in @(@{ n = 'default-on'; e = '' }, @{ n = 'off'; e = '0' })) {
    if ($setting.e -ne '') { $env:NINFER_TP2_AR_WATCHDOG = $setting.e }
    else { Remove-Item Env:NINFER_TP2_AR_WATCHDOG -ErrorAction SilentlyContinue }
    $out = "build-win/r87/inject-$($setting.n)"
    pwsh -NoProfile -Command "& './tools/tp_bootstrap/r72_stall_injection.ps1' -Calls @(700) -OutDir '$out'" 2>&1 |
        Select-String 'INJECT call' | ForEach-Object { $_.Line }
    $log    = Join-Path $out 'inject-700.log'
    $detail = @(Select-String -Path $log -Pattern '\[ar-watch\] calls=' -ErrorAction SilentlyContinue).Count
    $stall  = @(Select-String -Path $log -Pattern '\[ar-watch\] stalled at' -ErrorAction SilentlyContinue).Count
    Write-Output ("DUMP {0} stall_lines={1} detail_lines={2}" -f $setting.n, $stall, $detail)
    if ($detail -gt 0) {
        Select-String -Path $log -Pattern '\[ar-watch\] calls=' | Select-Object -First 1 | ForEach-Object { Write-Output ('  ' + $_.Line.Trim()) }
    }
}
Remove-Item Env:NINFER_TP2_AR_WATCHDOG -ErrorAction SilentlyContinue
Write-Output 'WATCHDOG_DEFAULT_CHECK_DONE'
