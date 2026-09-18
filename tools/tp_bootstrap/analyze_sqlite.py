import sqlite3
c = sqlite3.connect('/tmp/tp2_prof.sqlite')
K = 'CUPTI_ACTIVITY_KIND_KERNEL'
span_ns = c.execute('SELECT MAX(end)-MIN(start) FROM %s' % K).fetchone()[0]
min_ns  = c.execute('SELECT MIN(start) FROM %s' % K).fetchone()[0]
win = min_ns + span_ns - int(4.5e9)
row = c.execute('SELECT COUNT(*), SUM(end-start)/1e6, (MAX(end)-MIN(start))/1e6 FROM %s WHERE start >= ?' % K, (win,)).fetchone()
print('REQUEST WINDOW: instances=%d busy_ms=%.1f span_ms=%.1f' % row)
print('GPU busy fraction = %.1f%%' % (100.0*row[1]/row[2]))
print('--- per device (request window) ---')
for r in c.execute('SELECT deviceId, COUNT(*), SUM(end-start)/1e6 FROM %s WHERE start >= ? GROUP BY deviceId' % K, (win,)):
    print('dev%d: n=%d busy_ms=%.1f' % r)
print('--- top 14 kernels (request window) ---')
q = ('SELECT s.value, COUNT(*), SUM(k.end-k.start)/1e6 FROM %s k JOIN StringIds s ON k.demangledName=s.id ' % K)
q += 'WHERE k.start >= ? GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC LIMIT 14'
for name, n, ms in c.execute(q, (win,)):
    print('%9.1f ms  x%-6d %s' % (ms, n, name[:118]))
print('--- memcpy/memset (request window) ---')
for tbl in ('CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET'):
    try:
        r = c.execute('SELECT COUNT(*), SUM(end-start)/1e6 FROM %s WHERE start >= ?' % tbl, (win,)).fetchone()
        print('%s: n=%d busy_ms=%.1f' % (tbl, r[0], r[1]))
    except sqlite3.OperationalError:
        pass
print('--- synchronization (request window) ---')
try:
    r = c.execute('SELECT COUNT(*), SUM(end-start)/1e6 FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION WHERE start >= ?', (win,)).fetchone()
    print('sync: n=%d busy_ms=%.1f' % r)
except sqlite3.OperationalError:
    pass