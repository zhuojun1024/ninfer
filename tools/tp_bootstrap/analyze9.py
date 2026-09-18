import sqlite3
c = sqlite3.connect('/tmp/tp2_prof.sqlite')
for cand in ['CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_SYNCHRONIZATION','CUPTI_ACTIVITY_KIND_MEMCPY']:
    cols = [r[1] for r in c.execute('PRAGMA table_info("%s")' % cand).fetchall()]
    print(cand, 'cols:', cols)
    n = c.execute('SELECT COUNT(*) FROM "%s"' % cand).fetchone()[0]
    print('  rows:', n)
print()
# Runtime API: find cudaStreamSynchronize (name via StringIds? or column?). Inspect a sample.
rows = c.execute('SELECT * FROM "CUPTI_ACTIVITY_KIND_RUNTIME" LIMIT 3').fetchall()
print('sample runtime rows:', rows)
