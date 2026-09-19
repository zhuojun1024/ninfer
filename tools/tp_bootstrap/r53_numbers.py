#!/usr/bin/env python3
"""Round 53: read each fixture's own printed number back through the TP-2 vision path.

The load fixtures are labelled images, so the number in the reply is a content-specific answer that
a broken vision channel cannot produce. Thinking is left on and the budget is wide enough for the
final answer to land in the content channel.
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


def ask(path: str, question: str, max_tokens: int): 
    body = {
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url(path)}},
            {"type": "text", "text": question},
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
    }
    request = urllib.request.Request(SERVE + "/v1/chat/completions",
                                     data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(request, timeout=900) as response:
        return time.time() - started, json.loads(response.read().decode("utf-8"))


def main() -> int:
    for index in (0, 3, 7, 11):
        path = "%s/load_%02d.png" % (MEDIA, index)
        wall, body = ask(path, "What number is printed in the image? Reply with the digits only.",
                         320)
        if "choices" not in body:
            print("load_%02d: ERROR %s" % (index, str(body)[:300]))
            return 1
        message = body["choices"][0]["message"]
        content = (message.get("content") or "").strip()
        print("load_%02d expect=%d wall=%.1fs usage=%s" %
              (index, index, wall, json.dumps(body.get("usage"), ensure_ascii=False)))
        print("    answer=%r" % content[:120])
        print("    thinking=%r" % (message.get("reasoning_content") or "")[-160:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
