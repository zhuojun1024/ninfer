# Start ninfer-serve with the crashed run's production settings and replay a growing conversation.
#
#   pwsh -File tools/tp_bootstrap/r67_repro.ps1 -Model D:/LLM/<artifact>.ninfer -Tag q4all
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $Model,
    [Parameter(Mandatory = $true)][string] $Tag,
    [int] $TurnTokens = 12000,
    [int] $Turns = 5,
    [string] $Name = 'qwen3.8-27b-w4a4-w8a8-q4all',
    [string] $OutDir = 'build-win/r67'
)

$ErrorActionPreference = 'Stop'
Set-Location 'D:\Documents\workbench\ninfer'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$env:PATH = 'D:/ffmpeg-dev/expanded/ffmpeg-master-latest-win64-gpl-shared/bin;' +
            'D:/curl-dev/expanded/curl-8.22.0_1-win64-mingw/bin;' + $env:PATH
$binary = 'C:\ninfer\ninfer-serve.exe'
$log    = Join-Path $OutDir ("serve-" + $Tag + ".log")

Get-Process ninfer-serve -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 3
if (Test-Path $log) { Remove-Item $log -Force }

$cliArgs = @(
    $Model, '--devices', '0,1', '--max-context', '229376', '--port', '3456', '--host', '0.0.0.0',
    '--kv-dtype', 'k8v4', '--log-level', 'info', '--max-concurrency', '1', '--prefill-chunk', '1024',
    '--max-pending-requests', '16', '--host-state-slots', '32', '--host-kv-mib', '24576',
    '--max-private-continuations', '4', '--chat-template', 'D:/LLM/chat_template.jinja',
    '--spec', 'dflash2', '--draft-tokens', '7', '--lm-head-draft',
    '--reasoning-effort', 'medium', '--temperature', '0.7', '--top-k', '20', '--top-p', '0.8',
    '--vision', '--vision-item-tokens', '8192', '--preserve-thinking'
)
$proc = Start-Process -FilePath $binary -ArgumentList $cliArgs -NoNewWindow -PassThru -RedirectStandardError $log
$ready = $false
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 2
    if ($proc.HasExited) { break }
    try {
        if ((Invoke-WebRequest -Uri 'http://127.0.0.1:3456/health' -TimeoutSec 3 -UseBasicParsing).StatusCode -eq 200) { $ready = $true; break }
    } catch { }
}
if (-not $ready) {
    "SERVER_NOT_READY exited=$($proc.HasExited)"
    Get-Content $log -Tail 5 -ErrorAction SilentlyContinue
    if (-not $proc.HasExited) { $proc | Stop-Process -Force }
    exit 1
}
"SERVER_READY tag=$Tag"
python tools/tp_bootstrap/r67_longcontext_replay.py --model $Name --turn-tokens $TurnTokens --turns $Turns
$probe = $LASTEXITCODE
"PROBE_EXIT=$probe"
Start-Sleep -Seconds 2
"server_alive=$(-not $proc.HasExited)"
if (-not $proc.HasExited) { $proc | Stop-Process -Force }
"--- log tail ---"
Get-Content $log -Tail 12 -ErrorAction SilentlyContinue
