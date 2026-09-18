import sqlite3
c = sqlite3.connect('/tmp/tp2_prof.sqlite')
K = 'CUPTI_ACTIVITY_KIND_KERNEL'
q = ('SELECT k.start, k.end FROM %s k JOIN StringIds s ON k.demangledName=s.id ' % K)
q += "WHERE s.value LIKE '%residual_add_bf16x8%' AND k.deviceId=0 ORDER BY k.start"
ra = c.execute(q).fetchall()
forwards = []
i = 0
while i+64 <= len(ra):
    grp = ra[i:i+64]
    forwards.append((grp[0][0], grp[-1][1]))
    i += 64
# profile request = last 24 forwards (decode). Inter-forward gaps:
gaps = []
for j in range(len(forwards)-24, len(forwards)-1):
    g = forwards[j+1][0] - forwards[j][1]
    gaps.append(g/1e6)
print('inter-forward gaps (ms), profile decode window:')
for k,g in enumerate(gaps):
    print('  after fwd %d: %8.2f ms' % (k, g))
import statistics
print('mean %.1f ms, median %.1f ms, min %.1f, max %.1f' % (statistics.mean(gaps), statistics.median(gaps), min(gaps), max(gaps)))
print('=> wall/token = forward_span + gap')
