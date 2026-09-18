#!/bin/bash
head -c 900 /home/zhuojun/prof/smoke.json; echo
echo '--- keys ---'
python3 -c "import json;d=json.load(open('/home/zhuojun/prof/smoke.json'));c=d['choices'][0];print(sorted(c.keys()));print(sorted(c['message'].keys()));print('finish=',c.get('finish_reason'));print('usage=',d.get('usage'))"
