import sqlite3, re
c = sqlite3.connect('/tmp/tp2_prof.sqlite')
K = 'CUPTI_ACTIVITY_KIND_KERNEL'
def kern(dev):
    return c.execute('''SELECT s.value, k.start, k.end FROM %s k JOIN StringIds s ON k.demangledName=s.id WHERE k.deviceId=? ORDER BY k.start''' % K, (dev,)).fetchall()
for dev in (0, 1):
    rows = kern(dev)
    ar = [r for r in rows if 'ar_inplace_bf16' in r[0]]
    total_ms = sum(r[2]-r[1] for r in rows)/1e6
    ar_ms = sum(r[2]-r[1] for r in ar)/1e6
    print('dev%d: kernels=%d ar=%d total_gpu=%.1f ms ar_ms=%.1f' % (dev, len(rows), len(ar), total_ms, ar_ms))
# group AR calls into forwards
rows = kern(0)
ar = [r for r in rows if 'ar_inplace_bf16' in r[0]]
per = 128
n_for = len(ar)//per
print('AR calls=%d -> forwards=%d (per=%d)' % (len(ar), n_for, per))
bounds = []
for i in range(n_for):
    chunk = ar[i*per:(i+1)*per]
    bounds.append((chunk[0][1], chunk[-1][2]))
dec = bounds[-24:]
d0, d1 = dec[0][0], dec[-1][1]
span = (d1-d0)/1e6
print('decode window: %.1f ms over 24 tokens -> %.2f ms/token (wall)' % (span, span/24))
for dev in (0,1):
    q = '''SELECT s.value, COUNT(*), SUM(k.end-k.start)/1e6 FROM %s k JOIN StringIds s ON k.demangledName=s.id WHERE k.start>=? AND k.end<=? AND k.deviceId=? GROUP BY s.value ORDER BY 3 DESC''' % K
    agg = c.execute(q, (d0, d1, dev)).fetchall()
    tot = sum(a[2] for a in agg)
    busy = [r for r in rows]
    print()
    print('=== dev%d decode window: gpu_busy=%.2f ms/token, idle=%.2f ms/token' % (dev, tot/24, (span-tot)/24))
    for name,n,ms in agg[:14]:
        print('  %8.3f ms/tok x%-5d %s' % (ms/24, n/24, name[:95]))
    print('  %8.3f ms/tok  TOTAL %d distinct names' % (tot/24, len(agg)))
    gaps = 0.0
    rr = [r for r in rows if r[1]>=d0 and r[2]<=d1]
    for i in range(1,len(rr)):
        g = rr[i][1]-rr[i-1][2]
        if g>0: gaps += g
    print('  host-side gap total %.2f ms/token over %d gaps (mean %.1f us)' % (gaps/1e6/24, (len(rr)-1)/24, gaps/(len(rr)-1)/1000))