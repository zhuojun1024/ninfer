# Interpret a transport watchdog dump, so a failed acceptance arm leaves an actionable record.
#
# The watchdog prints the arrival and write-order slot arrays of both devices while the host-side
# collective counter is not advancing. A slot holds the rendezvous id a side published for one slice
# of a collective, and the arrays retain the last value written, so the *magnitude* of the A/B gap
# classifies the give-up:
#   - a side whose whole array is zero never reached its arrival write (its kernel never ran);
#   - an A/B gap of one id is an off-by-one between the two sides' numbering;
#   - a gap of a whole block (hundreds) means the two sides are running different id blocks, which is
#     what a captured graph reading a rendezvous cell that the host already re-armed looks like.
# One dump line as the watchdog writes it. Both the interpreter and the "is there evidence" test
# read this pattern, so the two cannot drift apart.
$ArWatchDumpPattern = '\[ar-watch\] calls=(\d+) last=(\d+) id=(\d+) arrA=\[([^\]]*)\] ordA=\[([^\]]*)\] arrB=\[([^\]]*)\] ordB=\[([^\]]*)\]'

# True when the log carries at least one dump. The watchdog writes one only when a device actually
# tripped the bounded spin, which is what makes this the test for "there is transport evidence here".
function Test-ArWatchDump {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string] $LogPath)

    if (-not (Test-Path $LogPath)) { return $false }
    return [bool](Select-String -Path $LogPath -ErrorAction SilentlyContinue -Pattern $ArWatchDumpPattern -Quiet)
}

function Get-ArWatchReport {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string] $LogPath)

    if (-not (Test-Path $LogPath)) { return "no log at $LogPath" }
    $dumps = @(Select-String -Path $LogPath -ErrorAction SilentlyContinue -Pattern $ArWatchDumpPattern)
    if ($dumps.Count -eq 0) {
        return "no ar-watch dump in the log: the watchdog was not enabled, or the process died before it reported"
    }
    $g = $dumps[-1].Matches[0].Groups
    $ra = @($g[4].Value.Split(" ") | Where-Object { $_ -ne "" })
    $rb = @($g[6].Value.Split(" ") | Where-Object { $_ -ne "" })
    $head = "last dump calls=" + $g[1].Value + " last_bytes=" + $g[2].Value + " id=" + $g[3].Value +
            " arrA=[" + ($ra -join " ") + "] arrB=[" + ($rb -join " ") + "]"

    $nonzeroA = @($ra | Where-Object { $_ -ne "0" })
    $nonzeroB = @($rb | Where-Object { $_ -ne "0" })
    if ($nonzeroA.Count -eq 0 -and $nonzeroB.Count -eq 0) {
        return $head + " | verdict: neither side has published anything yet"
    }
    if ($nonzeroA.Count -eq 0) { return $head + " | verdict: shard A NEVER PUBLISHED (shard B holds " + ($nonzeroB -join ",") + ")" }
    if ($nonzeroB.Count -eq 0) { return $head + " | verdict: shard B NEVER PUBLISHED (shard A holds " + ($nonzeroA -join ",") + ")" }

    $newestA = [uint64]$nonzeroA[0]
    $newestB = [uint64]$nonzeroB[0]
    $gap = [int64]$newestA - [int64]$newestB
    $kind = if ($gap -eq 0) {
        "both sides last published the same id " + $newestA + " (the skew, if any, is on a slot neither side reached)"
    } elseif ([Math]::Abs($gap) -eq 1) {
        "OFF-BY-ONE: A=" + $newestA + " B=" + $newestB + " - the two sides numbered the same collective differently"
    } elseif ([Math]::Abs($gap) -ge 64) {
        "BLOCK-LEVEL: A=" + $newestA + " B=" + $newestB + " (gap " + $gap + ") - the two sides are on different id blocks"
    } else {
        "SKEW: A=" + $newestA + " B=" + $newestB + " (gap " + $gap + ")"
    }
    return $head + " | verdict: " + $kind
}

# Copy a log aside under a name a re-run cannot overwrite, and write the interpreted dump beside it.
# Only a log that actually carries a dump is preserved. A failure without one - a test assertion, a
# bad artifact, a port clash - is already fully described by the ordinary run log, and copying it
# aside only scattered *-FAILED files that had to be cleaned up by hand.
function Save-ArWatchEvidence {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string] $LogPath,
        [Parameter(Mandatory = $true)][string] $Stem
    )
    if (-not (Test-ArWatchDump -LogPath $LogPath)) {
        Write-Host ("no transport evidence to preserve for " + $Stem)
        return $null
    }
    $kept = $Stem + "-FAILED.log"
    Copy-Item $LogPath $kept -Force
    $report = Get-ArWatchReport -LogPath $kept
    Set-Content -Path ($Stem + "-FAILED.arwatch.txt") -Value $report -Encoding UTF8
    Write-Host ("preserved transport evidence: " + $kept)
    Write-Host ("  " + $report)
    return $kept
}
