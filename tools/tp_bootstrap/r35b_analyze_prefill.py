import sqlite3, collections
db = sqlite3.connect('/home/zhuojun/prof/prefill_prof_c1024.sqlite')
strings = dict(db.execute('SELECT id, value FROM StringIds'))
rows = db.execute('SELECT start, end, shortName, demangledName, deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchall()
rows = [(s, e, e - s, strings.get(sn, ''), strings.get(dn, ''), dev) for s, e, sn, dn, dev in rows]
print('total kernel rows:', len(rows))
lo = min(r[0] for r in rows); hi = max(r[1] for r in rows)
span = (hi - lo) / 1e6
print('trace span ms: %.1f' % span)
dev_busy = collections.defaultdict(int)
for s, e, d, sn, dn, dev in rows:
    dev_busy[dev] += d
for dev, busy in sorted(dev_busy.items()):
    print('device %s: kernels busy %.1f ms of %.1f ms span (%.0f%%)' % (dev, busy / 1e6, span, 100 * busy / 1e6 / span))
ar = [(s, e, d) for s, e, d, sn, dn, dev in rows if 'ar_inplace' in sn]
print('AR instances total: %d, total %.1f ms' % (len(ar), sum(d for _, _, d in ar) / 1e6))
for a, b in [(0, 50e3), (50e3, 100e3), (100e3, 500e3), (500e3, 1e6), (1e6, 2e6), (2e6, 1e18)]:
    sel = [d for _, _, d in ar if a <= d < b]
    print('  %7.0f-%7.0f us: %5d instances, total %8.1f ms' % (a / 1e3, b / 1e3, len(sel), sum(sel) / 1e6))
win = hi - int(1.30e9)
wrows = [r for r in rows if r[0] >= win]
print('--- last 1.30 s: %d kernels ---' % len(wrows))
agg = collections.defaultdict(lambda: [0, 0])
for s, e, d, sn, dn, dev in wrows:
    agg[sn[:58]][0] += 1
    agg[sn[:58]][1] += d
for key, (cnt, tot) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:14]:
    print('  %8.1f ms  %5d x  %s' % (tot / 1e6, cnt, key))
war = [(s, e, d) for s, e, d, sn, dn, dev in wrows if 'ar_inplace' in sn]
if war:
    ds = sorted(d for _, _, d in war)
    print('window AR: %d instances, total %.1f ms, median %.1f us, p90 %.1f us, max %.1f us' % (
        len(war), sum(ds) / 1e6, ds[len(ds) // 2] / 1e3, ds[int(len(ds) * 0.9)] / 1e3, ds[-1] / 1e3))
# gaps in the last window per device (stream-agnostic): find the biggest idle gaps
for dev in sorted(set(r[5] for r in wrows)):
    dev_rows = sorted([r for r in wrows if r[5] == dev], key=lambda r: r[0])
    gaps = []
    cur_end = dev_rows[0][1]
    for s, e, d, sn, dn, _ in dev_rows[1:]:
        if s > cur_end:
            gaps.append((s - cur_end, cur_end))
        cur_end = max(cur_end, e)
    gaps.sort(reverse=True)
    print('device %s: biggest idle gaps (ms): %s' % (dev, ', '.join('%.2f' % (g / 1e6) for g, _ in gaps[:8])))
