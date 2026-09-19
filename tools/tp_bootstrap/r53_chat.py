#!/usr/bin/env python3
"""Round 53b: an image turn followed by a text turn in the same conversation.

A chat that carries an image keeps it in the history of every later turn, so the follow-up request
is still multimodal. This prints both replies and then the serve's own request lines, which carry
the TTFT and the decode rate the two paths actually reached.
"""
from __future__ import annotations

import base64
import json
import sys
import time
import urllib.request

SERVE = "http://127.0.0.1:8088"
MEDIA = "/mnt/d/Documents/workbench/ninfer/bench/fixtures/ttft/media"


def data_url(path: str) -> str:
    with open(path, "rb") as handle:
        return "data:image/png;base64," + base64.b64encode(handle.read()).decode("ascii")


def chat(messages, max_tokens=192):
    body = {"model": "qwen3.8-27b", "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.0, "top_k": 1, "top_p": 1.0}
    request = urllib.request.Request(SERVE + "/v1/chat/completions",
                                     data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(request, timeout=900) as response:
        return time.time() - started, json.loads(response.read().decode("utf-8"))


def text_of(body) -> str:
    message = body["choices"][0]["message"]
    return (message.get("content") or "") + (message.get("reasoning_content") or "")


def main() -> int:
    image = {"type": "image_url", "image_url": {"url": data_url("%s/load_07.png" % MEDIA)}}
    first_prompt = "What number is printed in the image? Reply with the digits only."
    wall1, body1 = chat([{"role": "user", "content": [image, {"type": "text", "text": first_prompt}]}])
    reply1 = text_of(body1)
    print("turn1 wall=%.2fs usage=%s" % (wall1, json.dumps(body1.get("usage"), ensure_ascii=False)))
    print("      reply=%r" % reply1[:160])

    follow_up = "Now multiply that number by 3 and reply with just the result."
    messages = [
        {"role": "user", "content": [image, {"type": "text", "text": first_prompt}]},
        {"role": "assistant", "content": reply1},
        {"role": "user", "content": follow_up},
    ]
    wall2, body2 = chat(messages)
    print("turn2 wall=%.2fs usage=%s" % (wall2, json.dumps(body2.get("usage"), ensure_ascii=False)))
    print("      reply=%r" % text_of(body2)[:160])
    if "choices" not in body1 or "choices" not in body2:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
