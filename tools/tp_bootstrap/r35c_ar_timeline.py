import sqlite3, collections
db = sqlite3.connect('/home/zhuojun/prof/prefill_prof_c1024.sqlite')
strings = dict(db.execute('SELECT id, value FROM StringIds'))
rows = []
for s, e, sn, gx, dev in db.execute('SELECT start, end, shortName, gridX, deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL'):
    rows.append((s, e, e - s, strings.get(sn, ''), gx, dev))
hi = max(r[1] for r in rows)
lo = min(r[0] for r in rows)
print('span %.0f ms' % ((hi - lo) / 1e6))
ar = sorted([r for r in rows if 'ar_inplace' in r[3]], key=lambda r: r[0])
print('AR total %d, by gridX: %s' % (len(ar), dict(collections.Counter(r[4] for r in ar))))
for gx in sorted(set(r[4] for r in ar)):
    sel = [r[2] for r in ar if r[4] == gx]
    sel.sort()
    print('  gridX=%2d: n=%4d total %7.1f ms  med %7.1f us  p90 %7.1f us  max %7.1f us' % (
        gx, len(sel), sum(sel)/1e6, sel[len(sel)//2]/1e3, sel[int(len(sel)*0.9)]/1e3, sel[-1]/1e3))
# chronological: per device, the biggest contiguous run of large-payload ARs
print('--- chronological AR durations (us) per device, first 140 ---')
for dev in sorted(set(r[5] for r in ar)):
    d = [r for r in ar if r[5] == dev]
    # find the first AR with gridX>1 (prefill) and print 40 around it
    idx = next((i for i, r in enumerate(d) if r[4] > 1), None)
    start = max(0, (idx or 0) - 5)
    seq = d[start:start + 60]
    print(' dev%s from index %d: %s' % (dev, start, ' '.join('%d' % (r[2]/1e3) for r in seq)))
    print('   gridX: %s' % ' '.join('%d' % r[4] for r in seq))
# gap analysis around a slow AR
d0 = [r for r in ar if r[5] == 0]
slow = [r for r in d0 if r[2] > 2e6]
print('dev0 slow ARs (>2ms): %d' % len(slow))
if slow:
    t = slow[len(slow)//2][0]
    near = sorted([r for r in rows if r[5] == 0 and r[0] >= t - 6e6 and r[0] <= t + 6e6], key=lambda r: r[0])
    print('  kernels around a slow dev0 AR (rel ms, us, name):')
    for r in near[:40]:
        print('   %+8.3f  %9.1f  %s' % ((r[0]-t)/1e6, r[2]/1e3, r[3][:60]))
