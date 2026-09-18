#!/bin/bash
cd /home/zhuojun/ninfer
echo '--- deployed source check ---'
grep -n 'kArMaxBlocks\|ar_inplace_bf16<<<\|auto\* mine_a' src/core/tp/device_pair.cu | head
echo '--- binary mtime ---'
ls -la build/apps/ninfer-serve
echo '--- gridX distribution of all kernels in the last trace ---'
cd /home/zhuojun/prof && python3 - <<'EOF'
import sqlite3, collections
db = sqlite3.connect('/home/zhuojun/prof/prefill_prof_c1024.sqlite')
strings = dict(db.execute('SELECT id, value FROM StringIds'))
c = collections.Counter()
for sn, gx in db.execute('SELECT shortName, gridX FROM CUPTI_ACTIVITY_KIND_KERNEL'):
    c[(strings.get(sn,'')[:40], gx)] += 1
for (name, gx), n in sorted(c.items(), key=lambda kv: -kv[1])[:12]:
    print('  gridX=%4d  n=%5d  %s' % (gx, n, name))
EOF
