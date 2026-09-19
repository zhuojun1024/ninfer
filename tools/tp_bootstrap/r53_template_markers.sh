#!/bin/bash
T=/home/zhuojun/models/chat_template.jinja
echo "--- size ---"; wc -c "$T"
echo "--- markers (count) ---"
for k in reasoning_effort enable_thinking preserve_thinking tools function image vision video 'im_start' 'think'; do
  printf '%-18s %s\n' "$k" "$(grep -c -- "$k" "$T")"
done
echo "--- first 40 lines ---"
head -40 "$T"
