#!/bin/bash
set -u
SW=/home/zhuojun/prof
res=$SW/r48_longctx.log
: > "$res"
python3 - <<'PY'
import json
base = json.load(open("/home/zhuojun/prof/essay_req.json"))
filler = "这是一段用于测试长上下文的中文材料，内容本身没有特殊含义，只用于填充提示长度。"
base["messages"][0]["content"] = "请阅读下面的材料，然后用中文写一段约200字的总结。\n\n" + filler * 5400
base["max_tokens"] = 220
json.dump(base, open("/home/zhuojun/prof/r48_long_req.json", "w"))
print("built prompt chars=%d keys=%s" % (len(base["messages"][0]["content"]), sorted(base.keys())))
PY
grep -a 'capacity |' "$SW/serve_supervised.log" | tail -1 >> "$res"
curl -s --max-time 900 http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
  --data-binary @/home/zhuojun/prof/r48_long_req.json -o "$SW/r48_long.json" \
  -w "long http=%{http_code} time_total=%{time_total}\n" >> "$res"
python3 - <<'PY' >> "$res"
import json, re
try:
    d = json.load(open("/home/zhuojun/prof/r48_long.json"))
    if "choices" not in d:
        print("response:", json.dumps(d)[:300])
    else:
        u = d.get("usage", {})
        content = d["choices"][0]["message"].get("content") or ""
        runs = [len(x.group(0)) for x in re.finditer(r"0+", content)]
        print("prompt_tokens=%s completion_tokens=%s content_chars=%d longest0=%d" % (
            u.get("prompt_tokens"), u.get("completion_tokens"), len(content), max(runs) if runs else 0))
        print("head:", content[:70].replace("\n", " "))
except Exception as exc:
    print("ERROR", exc)
PY
echo R48_DONE >> "$res"
