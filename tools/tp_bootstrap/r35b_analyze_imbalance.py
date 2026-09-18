import sqlite3, collections
db = sqlite3.connect('/home/zhuojun/prof/prefill_prof_c1024.sqlite')
strings = dict(db.execute('SELECT id, value FROM StringIds'))
rows = []
for s, e, sn, gx, dev in db.execute('SELECT start, end, shortName, gridX, deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL'):
    rows.append((s, e, e - s, strings.get(sn, ''), gx, dev))
hi = max(r[1] for r in rows)
win = hi - int(1.30e9)
w = [r for r in rows if r[0] >= win]
print('window kernels:', len(w), 'wall 1300 ms')
for dev in sorted(set(r[5] for r in w)):
    d = [r for r in w if r[5] == dev]
    ar = sum(r[2] for r in d if 'ar_inplace' in r[3])
    tot = sum(r[2] for r in d)
    print('device %s: kernel-busy %7.1f ms  (AR %7.1f ms, non-AR %7.1f ms)  idle %7.1f ms' % (
        dev, tot / 1e6, ar / 1e6, (tot - ar) / 1e6, 1.3 - tot / 1e6))
    ars = sorted(r[2] for r in d if 'ar_inplace' in r[3])
    if ars:
        print('   AR  n=%4d med %7.1f us p90 %7.1f us max %7.1f us' % (len(ars), ars[len(ars)//2]/1e3, ars[int(len(ars)*0.9)]/1e3, ars[-1]/1e3))
        print('   AR by gridX:', sorted(collections.Counter(r[4] for r in d if 'ar_inplace' in r[3]).items()))
# per-device aggregate for the heavy kernels
print('--- per-device per-kernel average duration (us) for the heavy kernels ---')
agg = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
for s, e, dur, sn, gx, dev in w:
    key = sn.split('<')[0].replace('void ninfer::ops::detail::', '').replace('ninfer::tp::<unnamed>::', '')
    a = agg[key][dev]
    a[0] += 1; a[1] += dur
for key, per in sorted(agg.items(), key=lambda kv: -sum(v[1] for v in kv[1].values()))[:8]:
    line = key[:44].ljust(44)
    for dev in sorted(per):
        cnt, tot = per[dev]
        line += '  dev%s: %5d x avg %8.1f us' % (dev, cnt, tot / cnt / 1e3)
    print(' ', line)
