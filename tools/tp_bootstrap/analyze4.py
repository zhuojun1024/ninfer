import sqlite3
c = sqlite3.connect('/tmp/tp2_prof.sqlite')
K = 'CUPTI_ACTIVITY_KIND_KERNEL'
# lm_head launches on dev0, sorted = one per forward (prefill+decode).
q = ('SELECT k.start, k.end FROM %s k JOIN StringIds s ON k.demangledName=s.id ' % K)
q += "WHERE s.value LIKE '%248320%5120%' AND k.deviceId=0"
lm = sorted(c.execute(q).fetchall())
print('total lm_head forwards (dev0):', len(lm))
# The LAST 24 are the decode tokens (request had max_tokens=24, output=24).
dec = lm[-24:]
d0 = dec[0][0]; d1 = dec[-1][1]
dec_span_ms = (d1-d0)/1e6
print('decode phase span: %.1f ms for 24 tokens => %.1f ms/token' % (dec_span_ms, dec_span_ms/24))

# busy time on each device within [d0, d1]
for dev in (0,1):
    r = c.execute('SELECT COUNT(*), SUM(end-start)/1e6 FROM %s WHERE start>=? AND end<=? AND deviceId=?' % K, (d0, d1, dev)).fetchone()
    print('dev%d decode: n=%d busy_ms=%.1f (busy frac %.1f%% of span)' % (dev, r[0], r[1], 100.0*r[1]/dec_span_ms))

# per-decode-token kernel breakdown (both devices), divided by 24
q2 = ('SELECT s.value, COUNT(*), SUM(k.end-k.start)/1e6 FROM %s k JOIN StringIds s ON k.demangledName=s.id ' % K)
q2 += 'WHERE k.start>=? AND k.end<=? GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC LIMIT 14'
tot=0
for name, n, ms in c.execute(q2, (d0, d1)):
    tot+=ms
    print('%8.3f ms/tok  x%-5d %s' % (ms/24, n, name[:95]))
print('SUM kernel ms/tok (both dev): %.2f' % (tot/24))