import sqlite3
c = sqlite3.connect('/tmp/tp2_prof.sqlite')
tabs = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print('tables:', tabs)
# find a string table
for t in tabs:
    try:
        cols = [r[1] for r in c.execute('PRAGMA table_info(%s)' % t)]
        if 'value' in cols or 'string' in cols or 'name' in cols:
            print('  ', t, cols[:6])
    except Exception as e:
        pass