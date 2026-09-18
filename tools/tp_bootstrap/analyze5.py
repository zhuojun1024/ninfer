import sqlite3
c = sqlite3.connect('/tmp/tp2_prof.sqlite')
K = 'CUPTI_ACTIVITY_KIND_KERNEL'
# residual_add is exactly 1 per layer per forward = 64 per forward (per device).
# Use it as the unambiguous forward marker on dev0.
q = ('SELECT k.start, k.end FROM %s k JOIN StringIds s ON k.demangledName=s.id ' % K)
q += "WHERE s.value LIKE '%residual_add_bf16x8%' AND k.deviceId=0 ORDER BY k.start"
ra = c.execute(q).fetchall()
print('total residual_add (dev0):', len(ra), '=> forwards:', len(ra)/64)

# Group into forwards: 64 consecutive residual_add = 1 forward.
# A forward boundary = gap > threshold between the 64th and next 1st.
forwards = []
i = 0
while i + 64 <= len(ra):
    grp = ra[i:i+64]
    forwards.append((grp[0][0], grp[-1][1]))
    i += 64
print('forwards detected:', len(forwards))

# The request = last 85 forwards (61 prefill + 24 decode). Warmup before.
# Identify decode = last 24 forwards.
dec = forwards[-24:]
d0 = dec[0][0]; d1 = dec[-1][1]
dec_ms = (d1-d0)/1e6
print('DECODE 24 forwards: span %.1f ms => %.2f ms/token => %.1f tok/s' % (dec_ms, dec_ms/24, 24000.0/dec_ms))

pre = forwards[-85:-24]
p0 = pre[0][0]; p1 = pre[-1][1]
pre_ms = (p1-p0)/1e6
print('PREFILL 61 forwards: span %.1f ms => %.2f ms/token' % (pre_ms, pre_ms/61))

# decode per-device busy + per-token kernel breakdown
for dev in (0,1):
    r = c.execute('SELECT COUNT(*), SUM(end-start)/1e6 FROM %s WHERE start>=? AND end<=? AND deviceId=?' % K, (d0, d1, dev)).fetchone()
    print('dev%d decode: n=%d busy_ms=%.1f => %.2f ms/token' % (dev, r[0], r[1], r[1]/24))
print('--- per-decode-token kernel (both dev, /24) ---')
q2 = ('SELECT s.value, COUNT(*), SUM(k.end-k.start)/1e6 FROM %s k JOIN StringIds s ON k.demangledName=s.id ' % K)
q2 += 'WHERE k.start>=? AND end<=? GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC LIMIT 12'
for name, n, ms in c.execute(q2, (d0, d1)):
    print('%8.3f ms/tok  x%-5d %s' % (ms/24, n, name[:90]))