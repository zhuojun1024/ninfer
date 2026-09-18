#!/bin/bash
ls -la /tmp/tp2_prof*
python3 - <<'EOF'
import sqlite3
c = sqlite3.connect('/tmp/tp2_prof.sqlite')
names = [r[0] for r in c.execute("select name from sqlite_master where type='table'").fetchall()]
print(len(names), 'tables')
print([n for n in names if 'KERNEL' in n or 'String' in n])
EOF