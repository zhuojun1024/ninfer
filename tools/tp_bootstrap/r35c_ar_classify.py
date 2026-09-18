import sqlite3, collections
db = sqlite3.connect('/home/zhuojun/prof/prefill_prof_c1024.sqlite')
strings = dict(db.execute('SELECT id, value FROM StringIds'))
rows = []
for s, e, sn, gx, dev in db.execute('SELECT start, end, shortName, gridX, deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL'):
    rows.append((s, e, e - s, strings.get(sn, ''), gx, dev))
bydev = collections.defaultdict(list)
for r in rows:
    bydev[r[5]].append(r)
for d in bydev:
    bydev[d].sort(key=lambda r: r[0])

def classify(dev, t):
    # nearest preceding non-AR kernel within 30 ms
    best = None
    for r in bydev[dev]:
        if r[0] >= t:
            break
        if 'ar_inplace' in r[3]:
            continue
        if t - r[1] < 30e6:
            best = r[3]
    if best is None:
        return 'none'
    if 'gemv' in best or 'sample' in best or 'lm_head' in best:
        return 'decode-ish'
    if 'mma' in best or 'silu' in best or 'causal_conv' in best or 'state_passing' in best:
        return 'prefill-ish'
    return 'other:' + best[:28]

groups = collections.defaultdict(list)
for s, e, dur, sn, gx, dev in rows:
    if 'ar_inplace' not in sn:
        continue
    groups[(dev, classify(dev, s))].append((dur, gx))
for key in sorted(groups):
    ds = sorted(x[0] for x in groups[key])
    gx = collections.Counter(x[1] for x in groups[key])
    print('dev%s %-14s n=%4d total %7.1f ms  med %7.1f us p90 %7.1f us max %7.1f us  gridX=%s' % (
        key[0], key[1], len(ds), sum(ds)/1e6, ds[len(ds)//2]/1e3, ds[int(len(ds)*0.9)]/1e3, ds[-1]/1e3, dict(gx)))
