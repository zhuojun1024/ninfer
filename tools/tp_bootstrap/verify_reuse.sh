#!/bin/bash
URL=http://127.0.0.1:8088/v1/chat/completions
M=qwen3.8-27b
run() {
  local name="$1"; local payload="$2"; local out=/tmp/out_$name.json
  printf '%s' "$payload" > /tmp/p_$name.json
  curl -s -X POST "$URL" -H 'Content-Type: application/json' --data-binary @/tmp/p_$name.json \
    -o "$out" -w "$name HTTP=%{http_code} wall=%{time_total}s\n"
}
S='You are a terse assistant.'
U1='Name the first planet from the sun.'
U2='Now name the second planet from the sun.'
U3='What color is the sky on a clear day?'
mk() { # $1 = extra assistant+user suffix or empty
  printf '{"model":"%s","messages":[{"role":"system","content":"%s"},{"role":"user","content":"%s"}%s],"max_tokens":24,"temperature":0,"stream":false}' "$M" "$S" "$U1" "$1"
}
P1=$(mk '')
P2=$(mk ",{\"role\":\"assistant\",\"content\":\"Mercury\"},{\"role\":\"user\",\"content\":\"$U2\"}")
PX=$(printf '{"model":"%s","messages":[{"role":"system","content":"%s"},{"role":"user","content":"%s"}],"max_tokens":8,"temperature":0,"stream":false}' "$M" "$S" "$U3")
echo '=== A: P1 (full prefill) ==='
run A "$P1"
echo '=== B: P2 (should reuse P1 prefix) ==='
run B "$P2"
echo '=== X: unrelated (invalidates cache) ==='
run X "$PX"
echo '=== B2: P2 again (full prefill) ==='
run B2 "$P2"
python3 - <<'EOF'
import json
def msg(p):
    d = json.load(open(p))
    m = d['choices'][0]['message']
    return m.get('content',''), m.get('reasoning_content','') or '', d.get('usage',{})
b  = msg('/tmp/out_B.json')
b2 = msg('/tmp/out_B2.json')
print('B  content=%r reasoning=%r' % (b[0], b[1][:200]))
print('B2 content=%r reasoning=%r' % (b2[0], b2[1][:200]))
print('MATCH' if b[0]==b2[0] and b[1]==b2[1] else 'MISMATCH')
EOF