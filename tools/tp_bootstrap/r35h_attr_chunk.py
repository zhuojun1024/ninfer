import sqlite3, collections
db = sqlite3.connect('/home/zhuojun/prof/prefill_prof_c1024.sqlite')
strings = dict(db.execute('SELECT id, value FROM StringIds'))
rows = []
for s, e, sn, gx, dev in db.execute('SELECT start, end, shortName, gridX, deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL'):
    rows.append((s, e, e - s, strings.get(sn, ''), gx, dev))   # 0 start 1 end 2 dur 3 name 4 gridX 5 dev
rows.sort(key=lambda r: r[0])
marks = [r[0] for r in rows if r[5] == 0 and r[4] == 2560 and 'residual_add' in r[3]]
print('wide-chunk residual_add marks: %d' % len(marks))
lo, hi = marks[0] - int(4e6), marks[-1] + int(6e6)
sel = [r for r in rows if lo <= r[0] <= hi and r[5] == 0]
wall = (hi - lo) / 1e6
agg = collections.defaultdict(lambda: [0, 0])
for r in sel:
    key = r[3]
    for pre in ('void ninfer::ops::detail::', 'ninfer::ops::', 'ninfer::tp::<unnamed>::'):
        key = key.replace(pre, '')
    key = key.split('(')[0]
    if 'kernel<' in key:
        key = key.split('<')[0]
    agg[key][0] += 1
    agg[key][1] += r[2]
total = sum(v[1] for v in agg.values())
print('dev0 window %.0f ms, kernels %d, busy %.0f ms (%.0f%%), idle %.0f ms' % (
    wall, len(sel), total / 1e6, 100 * total / 1e6 / wall, wall - total / 1e6))
print('%-46s %6s %10s %9s %8s' % ('kernel', 'n', 'total ms', 'avg us', '% busy'))
for name, (n, tot) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:18]:
    print('%-46s %6d %10.1f %9.1f %7.1f%%' % (name[:46], n, tot / 1e6, tot / n / 1e3, 100 * tot / 1e6 / wall))
