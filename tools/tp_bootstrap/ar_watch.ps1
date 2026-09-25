# Interpret a transport watchdog dump, so a failed acceptance arm leaves an actionable record.
#
# The watchdog prints the arrival and write-order slot arrays of both devices while the host-side
# collective counter is not advancing. A slot holds the rendezvous id each side published for one
# slice of one collective, so comparing the two arrays separates the two possible causes of a
# bounded-spin give-up: a token divergence (the two sides numbered the same collective differently,
# which is a protocol or timing defect) versus a side that never published at all (its kernel was
# late or its store was lost).
function Get-ArWatchReport {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string] $LogPath)

    if (-not (Test-Path $LogPath)) { return "no log at $LogPath" }
    $pattern = '\[ar-watch\] calls=(\d+) last=(\d+) id=(\d+) arrA=\[([^\]]*)\] ordA=\[([^\]]*)\] arrB=\[([^\]]*)\] ordB=\[([^\]]*)\]'
    $dumps = @(Select-String -Path $LogPath -ErrorAction SilentlyContinue -Pattern $pattern)
    if ($dumps.Count -eq 0) {
        return "no ar-watch dump in the log: the watchdog was not enabled, or the process died before it reported"
    }
    $g = $dumps[-1].Matches[0].Groups
    $a = @($g[4].Value.Split(" ") | Where-Object { $_ -ne "" })
    $b = @($g[6].Value.Split(" ") | Where-Object { $_ -ne "" })
    $head = "last dump calls=" + $g[1].Value + " last_bytes=" + $g[2].Value + " id=" + $g[3].Value +
            " arrA=[" + ($a -join " ") + "] arrB=[" + ($b -join " ") + "]"
    $verdicts = @()
    for ($i = 0; $i -lt [Math]::Min($a.Count, $b.Count); $i++) {
        if ($a[$i] -eq $b[$i]) { continue }
        if ($a[$i] -eq "0") { $verdicts += "slice " + $i + ": shard A never published (shard B holds " + $b[$i] + ")" }
        elseif ($b[$i] -eq "0") { $verdicts += "slice " + $i + ": shard B never published (shard A holds " + $a[$i] + ")" }
        else { $verdicts += "slice " + $i + ": TOKEN DIVERGENCE A=" + $a[$i] + " B=" + $b[$i] }
    }
    if ($verdicts.Count -eq 0) {
        return $head + " | verdict: both sides agree on every published slot, so the give-up is on an id neither side reached (a late peer, not a numbering defect)"
    }
    return $head + " | verdict: " + ($verdicts -join "; ")
}

# Copy a log aside under a name a re-run cannot overwrite, and write the interpreted dump beside it.
function Save-ArWatchEvidence {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string] $LogPath,
        [Parameter(Mandatory = $true)][string] $Stem
    )
    if (-not (Test-Path $LogPath)) { return $null }
    $kept = $Stem + "-FAILED.log"
    Copy-Item $LogPath $kept -Force
    $report = Get-ArWatchReport -LogPath $kept
    Set-Content -Path ($Stem + "-FAILED.arwatch.txt") -Value $report -Encoding UTF8
    Write-Host ("preserved transport evidence: " + $kept)
    Write-Host ("  " + $report)
    return $kept
}
