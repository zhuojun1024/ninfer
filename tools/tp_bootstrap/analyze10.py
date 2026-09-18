import sqlite3
c = sqlite3.connect('/tmp/tp2_prof.sqlite')
K = 'CUPTI_ACTIVITY_KIND_KERNEL'
q = ('SELECT k.start, k.end FROM %s k JOIN StringIds s ON k.demangledName=s.id ' % K)
q += "WHERE s.value LIKE '%residual_add_bf16x8%' AND k.deviceId=0 ORDER BY k.start"
ra = c.execute(q).fetchall()
forwards=[]; i=0
while i+64<=len(ra):
    forwards.append((ra[i][0], ra[i+63][1])); i+=64
dec=forwards[-24:]; d0=dec[0][0]; d1=dec[-1][1]
print('decode window: %.1f ms' % ((d1-d0)/1e6))
# host-side cudaStreamSynchronize (both plain and _v3020 suffixed names)
rows = c.execute('''SELECT s.value, COUNT(*), COALESCE(SUM(r.end-r.start),0)/1e6 FROM "CUPTI_ACTIVITY_KIND_RUNTIME" r JOIN StringIds s ON r.nameId=s.id WHERE r.start>=? AND r.end<=? GROUP BY s.value ORDER BY 3 DESC LIMIT 12''', (d0,d1)).fetchall()
print('top runtime APIs in decode window (host-side):')
for name,n,ms in rows:
    print('  %8.2f ms/tok  x%-6d %s' % (ms/24, n, name))
tot = c.execute('SELECT COALESCE(SUM(end-start),0)/1e6 FROM "CUPTI_ACTIVITY_KIND_RUNTIME" WHERE start>=? AND end<=?', (d0,d1)).fetchone()[0]
print('ALL runtime API host time: %.1f ms => %.2f ms/token' % (tot, tot/24))
# SYNCHRONIZATION activity (device-side sync records)
r = c.execute('SELECT COUNT(*), COALESCE(SUM(end-start),0)/1e6 FROM "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION" WHERE start>=? AND end<=?', (d0,d1)).fetchone()
print('SYNCHRONIZATION activity: n=%d total=%.1f ms => %.2f ms/token' % (r[0], r[1], r[1]/24))
# MEMCPY device-side time in window
r = c.execute('SELECT COUNT(*), COALESCE(SUM(end-start),0)/1e6 FROM "CUPTI_ACTIVITY_KIND_MEMCPY" WHERE start>=? AND end<=?', (d0,d1)).fetchone()
print('MEMCPY device time: n=%d total=%.1f ms => %.2f ms/token' % (r[0], r[1], r[1]/24))
# kernel busy (both devs) in window
for dev in (0,1):
    r = c.execute('SELECT COALESCE(SUM(end-start),0)/1e6 FROM %s WHERE start>=? AND end<=? AND deviceId=?' % K, (d0,d1,dev)).fetchone()
    print('dev%d kernel busy: %.1f ms => %.2f ms/token' % (dev, r[0], r[0]/24))
