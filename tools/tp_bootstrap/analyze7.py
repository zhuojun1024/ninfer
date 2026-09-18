import sqlite3
c = sqlite3.connect('/tmp/tp2_prof.sqlite')
K = 'CUPTI_ACTIVITY_KIND_KERNEL'
q = ('SELECT k.start, k.end FROM %s k JOIN StringIds s ON k.demangledName=s.id ' % K)
q += "WHERE s.value LIKE '%residual_add_bf16x8%' AND k.deviceId=0 ORDER BY k.start"
ra = c.execute(q).fetchall()
nf = len(ra)//64
print('total residual_add:', len(ra), '=> forwards:', nf)
forwards = []
i = 0
while i+64 <= len(ra):
    grp = ra[i:i+64]
    forwards.append((grp[0][0], grp[-1][1]))
    i += 64
for idx,(s,e) in enumerate(forwards):
    span=(e-s)/1e6
    tag = 'PREFILL' if span>80 else 'decode'
    print('%3d %s span %9.2f ms' % (idx, tag, span))
