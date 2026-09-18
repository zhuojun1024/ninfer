#!/bin/bash
echo '--- fresh prefill output (req#2, streaming) ---'
python3 - <<'PY'
import json, re
text = open('/tmp/long1.sse', encoding='utf-8', errors='replace').read()
reason, content = [], []
for line in text.splitlines():
    if not line.startswith('data: '):
        continue
    body = line[6:].strip()
    if body == '[DONE]':
        continue
    try:
        delta = json.loads(body)['choices'][0].get('delta', {})
    except Exception:
        continue
    reason.append(delta.get('reasoning_content') or '')
    content.append(delta.get('content') or '')
print('REASON:', repr(''.join(reason))[:600])
print('CONTENT:', repr(''.join(content))[:300])
PY
echo '--- reuse output (req#3, non-stream) ---'
python3 - <<'PY'
import json
d = json.load(open('/tmp/long2.json.out', encoding='utf-8'))
m = d['choices'][0]['message']
print('REASON:', repr(m.get('reasoning_content'))[:600])
print('CONTENT:', repr(m.get('content'))[:300])
PY
echo '--- serve log head ---'
head -14 /home/zhuojun/prof/serve_r35.log
echo '--- serve processes ---'
pgrep -a -f 'build/apps/ninfer'
