import sqlite3, collections
db = sqlite3.connect('/home/zhuojun/prof/prefill_prof_c1024.sqlite')
strings = dict(db.execute('SELECT id, value FROM StringIds'))
rows = []
for s, e, sn, gx, bx, dev in db.execute('SELECT start, end, shortName, gridX, blockX, deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL'):
    rows.append((s, e, e - s, strings.get(sn, ''), gx, bx, dev))
hi = max(r[1] for r in rows)
t0 = hi - int(1000e6)
def win(a, b, dev=0):
    sel = [r for r in rows if r[6] == dev and r[0] >= t0 + int(a*1e6) and r[0] < t0 + int(b*1e6)]
    c = collections.Counter()
    for r in sel:
        key = r[3].split('(')[0]
        key = key.replace('void ninfer::ops::detail::', '').replace('ninfer::tp::<unnamed>::', '').replace('void ninfer::ops::', '')
        c[(key[:46], r[4], r[5])] += 1
    print('--- dev%d [%d,%d) ms: %d kernels ---' % (dev, a, b, len(sel)))
    for (name, gx, bx), n in sorted(c.items(), key=lambda kv: -kv[1])[:10]:
        print('    %5d x  gridX=%-5d blockX=%-5d %s' % (n, gx, bx, name))
win(20, 700)
win(700, 860)
win(860, 1000)
