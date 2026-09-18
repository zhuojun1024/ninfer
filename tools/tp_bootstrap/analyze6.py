import sqlite3
c = sqlite3.connect('/tmp/tp2_prof.sqlite')
K = 'CUPTI_ACTIVITY_KIND_KERNEL'
# decode window from residual_add grouping (last 24 forwards on dev0)
q = ('SELECT k.start, k.end FROM %s k JOIN StringIds s ON k.demangledName=s.id ' % K)
q += "WHERE s.value LIKE '%residual_add_bf16x8%' AND k.deviceId=0 ORDER BY k.start"
ra = c.execute(q).fetchall()
forwards=[]
i=0
while i+64<=len(ra):
    forwards.append((ra[i][0], ra[i+63][1])); i+=64
dec=forwards[-24:]; d0=dec[0][0]; d1=dec[-1][1]

# all dev0 kernels in decode window, sorted
ker = c.execute('SELECT start,end FROM %s WHERE start>=? AND end<=? AND deviceId=0 ORDER BY start' % K, (d0,d1)).fetchall()
busy=sum(e-s for s,e in ker)
span=ker[-1][1]-ker[0][0]
gaps=[]
for i in range(1,len(ker)):
    g=ker[i][0]-ker[i-1][1]
    if g>0: gaps.append(g)
gaps.sort()
import statistics
tot_gap=sum(gaps)
print('decode dev0: %d kernels, span %.1f ms, busy %.1f ms, gap %.1f ms' % (len(ker), span/1e6, busy/1e6, tot_gap/1e6))
print('gap: mean %.1f us, median %.1f us, p90 %.1f us, max %.1f us, n_gaps %d' % (statistics.mean(gaps)/1e3, statistics.median(gaps)/1e3, gaps[int(0.9*len(gaps))]/1e3, max(gaps)/1e3, len(gaps)))
# how many gaps are 'large' (>50us = a real barrier/sync, not just launch)
big=[g for g in gaps if g>50000]
print('gaps >50us: %d totaling %.1f ms (avg %.1f us)' % (len(big), sum(big)/1e6, (sum(big)/len(big))/1e3 if big else 0))
small=[g for g in gaps if g<=50000]
print('gaps <=50us: %d totaling %.1f ms (avg %.1f us)' % (len(small), sum(small)/1e6, (sum(small)/len(small))/1e3 if small else 0))