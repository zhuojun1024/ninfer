import sqlite3
db = sqlite3.connect('/home/zhuojun/prof/prefill_prof_c1024.sqlite')
strings = dict(db.execute('SELECT id, value FROM StringIds'))
rows = []
for s, e, sn, gx, bx, dev in db.execute('SELECT start, end, shortName, gridX, blockX, deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL'):
    rows.append((s, e, e - s, strings.get(sn, ''), gx, bx, dev))
hi = max(r[1] for r in rows)
lo = hi - int(1000e6)
ar = sorted([r for r in rows if 'ar_inplace' in r[3] and r[0] >= lo], key=lambda r: r[0])
print('AR instances in the last 1000 ms: dev0=%d dev1=%d' % (
    len([r for r in ar if r[6] == 0]), len([r for r in ar if r[6] == 1])))
for dev in (0, 1):
    seq = [r for r in ar if r[6] == dev]
    print('--- dev%d: %d instances, elapsed_ms:duration_us (gridX/blockX) for the first 130 ---' % (dev, len(seq)))
    out = []
    for i, r in enumerate(seq[:130]):
        out.append('%d:%.0f(%d/%d)' % ((r[0] - lo) / 1e6, r[2] / 1e3, r[4], r[5]))
    for i in range(0, len(out), 8):
        print('   ' + ' '.join(out[i:i+8]))
