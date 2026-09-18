import sqlite3
c = sqlite3.connect('/home/zhuojun/prof/tp2.sqlite')
K = 'CUPTI_ACTIVITY_KIND_KERNEL'
rows = c.execute('''SELECT s.value, k.start, k.end FROM %s k JOIN StringIds s ON k.demangledName=s.id WHERE k.deviceId=0 ORDER BY k.start''' % K).fetchall()
ar = [r for r in rows if 'ar_inplace_bf16' in r[0]]
per = 128
n_for = len(ar)//per
bounds = [(ar[i*per][1], ar[(i+1)*per-1][2]) for i in range(n_for)]
dec = bounds[-24:]
d0, d1 = dec[0][0], dec[-1][1]
print('window %.2f ms, forwards=%d' % ((d1-d0)/1e6, n_for))
q = '''SELECT s.value, COUNT(*), SUM(k.end-k.start)/1e6 FROM %s k JOIN StringIds s ON k.demangledName=s.id WHERE k.start>=? AND k.end<=? AND k.deviceId=0 GROUP BY s.value ORDER BY 3 DESC''' % K
agg = c.execute(q, (d0, d1)).fetchall()
tot = sum(a[2] for a in agg)
print('%d distinct names, gpu busy %.3f ms/tok' % (len(agg), tot/24))
for name,n,ms in agg:
    print('%9.3f ms/tok  n=%4d  %.1f us/call  %s' % (ms/24, n, ms*1000/n, name))