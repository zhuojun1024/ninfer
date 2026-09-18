import sqlite3
c = sqlite3.connect('/tmp/tp2_prof.sqlite')
K = 'CUPTI_ACTIVITY_KIND_KERNEL'
span_ns = c.execute('SELECT MAX(end)-MIN(start) FROM %s' % K).fetchone()[0]
min_ns  = c.execute('SELECT MIN(start) FROM %s' % K).fetchone()[0]
win = min_ns + span_ns - int(4.5e9)

q = ('SELECT k.start, k.end FROM %s k JOIN StringIds s ON k.demangledName=s.id ' % K)
q += "WHERE k.start >= ? AND k.deviceId=0 AND s.value LIKE '%248320%5120%'"
lm = c.execute(q, (win,)).fetchall()
lm.sort()
print('lm_head launches (dev0, window):', len(lm))

dev0 = c.execute('SELECT start,end FROM %s WHERE start>=? AND deviceId=0 ORDER BY start' % K, (win,)).fetchall()
busy0 = sum(e-s for s,e in dev0)
print('dev0 busy in window: %.1f ms over %d kernels' % (busy0/1e6, len(dev0)))

gaps = 0
for i in range(1, len(dev0)):
    gap = dev0[i][0] - dev0[i-1][1]
    if gap > 0: gaps += gap
span0 = dev0[-1][1] - dev0[0][0]
print('dev0 span: %.1f ms, busy: %.1f ms, gaps: %.1f ms (busy frac %.1f%%)' % (span0/1e6, busy0/1e6, gaps/1e6, 100.0*busy0/span0))

calls = 1
for i in range(1, len(lm)):
    if lm[i][0] - lm[i-1][1] > 200000:
        calls += 1
print('lm_head calls (grouped):', calls, ' launches/call: %.1f' % (len(lm)/max(calls,1)))
lm_busy = sum(e-s for s,e in lm)
print('lm_head total busy (dev0): %.1f ms, per call: %.2f ms' % (lm_busy/1e6, lm_busy/1e6/max(calls,1)))