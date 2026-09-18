import sqlite3, re
c = sqlite3.connect('/tmp/tp2_prof.sqlite')
K = 'CUPTI_ACTIVITY_KIND_KERNEL'
q = ('SELECT k.start, k.end FROM %s k JOIN StringIds s ON k.demangledName=s.id ' % K)
q += "WHERE s.value LIKE '%residual_add_bf16x8%' AND k.deviceId=0 ORDER BY k.start"
ra = c.execute(q).fetchall()
forwards=[]; i=0
while i+64<=len(ra):
    forwards.append((ra[i][0], ra[i+63][1])); i+=64
dec=forwards[-24:]; d0=dec[0][0]; d1=dec[-1][1]
# per-device per-kernel time in decode window
rows = c.execute('''SELECT s.value, COUNT(*), SUM(k.end-k.start)/1e6 FROM %s k JOIN StringIds s ON k.demangledName=s.id WHERE k.start>=? AND k.end<=? AND k.deviceId=0 GROUP BY s.value ORDER BY 3 DESC LIMIT 25''' % K, (d0,d1)).fetchall()
print('PER-DEVICE (dev0) per-token kernel time, decode window:')
for name,n,ms in rows:
    print('  %8.3f ms/tok  x%-4d %s' % (ms/24, n/24, name[:100]))
# parse shapes for bandwidth estimate
def parse(name):
    m = re.search(r'(\d+)x(\d+)', name)
    if not m: return None
    return int(m.group(1)), int(m.group(2))
def dtype_bytes(name):
    if 'nvfp4' in name: return 0.5625
    if 'fp8' in name: return 1.0
    return 2.0
print()
print('effective bandwidth (dev0, per token):')
for name,n,ms in rows:
    p = parse(name)
    if not p or ms/24 < 0.05: continue
    a,b = p
    elems = a*b
    byts = elems*dtype_bytes(name)
    t = ms/24/1000.0  # seconds
    gbs = byts/t/1e9 if t>0 else 0
    print('  %8.1f GB/s  %8.3f ms/tok  %dx%d  %s' % (gbs, ms/24, a, b, name[:70]))
