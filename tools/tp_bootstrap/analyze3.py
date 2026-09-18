import sqlite3
c = sqlite3.connect('/tmp/tp2_prof.sqlite')
K = 'CUPTI_ACTIVITY_KIND_KERNEL'
min_ns  = c.execute('SELECT MIN(start) FROM %s' % K).fetchone()[0]
max_ns  = c.execute('SELECT MAX(end)   FROM %s' % K).fetchone()[0]
span_ms = (max_ns-min_ns)/1e6
# nsys profile ended ~22:47:17.8 (SIGINT after request+2s drain).
# So max_ns ~= 22:47:17.8 => min_ns ~= 22:47:10.3.
# Request: start 22:47:11.712, end 22:47:15.832 (serve log).
req_start = min_ns + int(1.412e9)   # 11.712 - 10.300
req_end   = min_ns + int(5.532e9)   # 15.832 - 10.300

# count lm_head in exact request window
q = ('SELECT COUNT(*) FROM %s k JOIN StringIds s ON k.demangledName=s.id ' % K)
q += "WHERE k.start>=? AND k.start<? AND s.value LIKE '%248320%5120%'"
n_lm = c.execute(q, (req_start, req_end)).fetchone()[0]
print('lm_head calls in request window:', n_lm, '(expect ~25 = 1 prefill + 24 decode)')

# per-device busy + span in request window
for dev in (0,1):
    r = c.execute('SELECT COUNT(*), SUM(end-start)/1e6, (MAX(end)-MIN(start))/1e6 FROM %s WHERE start>=? AND end<=? AND deviceId=?' % K, (req_start, req_end, dev)).fetchone()
    print('dev%d: n=%d busy_ms=%.1f span_ms=%.1f busy_frac=%.1f%%' % (dev, r[0], r[1], r[2], 100.0*r[1]/r[2]))

# per-token = busy / n_lm (decode tokens; prefill negligible)
n_tok = max(n_lm-1, 1)
print('--- per-token kernel time (request window, both devices) ---')
q2 = ('SELECT s.value, COUNT(*), SUM(k.end-k.start)/1e6 FROM %s k JOIN StringIds s ON k.demangledName=s.id ' % K)
q2 += 'WHERE k.start>=? AND k.end<=? GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC LIMIT 14'
for name, n, ms in c.execute(q2, (req_start, req_end)):
    print('%8.3f ms/tok  x%-5d %s' % (ms/n_tok, n, name[:100]))