#!/usr/bin/env python3
"""Round 53: TP-2 Vision smoke and memory verdict.

Sends real PNG fixtures to the running serve as OpenAI image_url data URLs, samples both inference
cards' memory while each request is in flight, and prints the reply plus the request accounting.
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
import threading
import time
import urllib.request

SERVE = "http://127.0.0.1:8088"
MEDIA = "/mnt/d/Documents/workbench/ninfer/bench/fixtures/ttft/media"
MODEL = "qwen3.8-27b"
CARDS = ("0", "2")


def data_url(path: str) -> str:
    with open(path, "rb") as handle:
        return "data:image/png;base64," + base64.b64encode(handle.read()).decode("ascii")


class Sampler(threading.Thread):
    """Peak memory.used per card while a request runs (nvidia-smi is authoritative on WSL2)."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.peak: dict[str, int] = {}
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                text = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout
                for line in text.splitlines():
                    index, used = (part.strip() for part in line.split(","))
                    if index in CARDS:
                        self.peak[index] = max(self.peak.get(index, 0), int(used))
            except Exception:
                pass
            self.stop.wait(0.05)


def ask(path: str, question: str, max_tokens: int) -> tuple[float, dict]:
    body = {
        "model": MODEL,
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
    question = "Describe this image in one short sentence."
    images = sys.argv[1:] or ["%s/load_%02d.png" % (MEDIA, index) for index in range(3)]
    for path in images:
        sampler = Sampler()
        sampler.start()
        try:
            wall, body = ask(path, question, 64)
        finally:
            sampler.stop.set()
            sampler.join(timeout=5)
        if "choices" not in body:
            print("%s: ERROR %s" % (path, str(body)[:400]))
            return 1
        choice = body["choices"][0]
        message = choice["message"]
        text = (message.get("content") or "") + "||" + (message.get("reasoning_content") or "")
        print("--- %s" % path)
        print("    wall=%.2fs finish=%s" % (wall, choice.get("finish_reason")))
        print("    usage=%s" % json.dumps(body.get("usage"), ensure_ascii=False))
        print("    peak_mib=%s" % json.dumps(sampler.peak))
        print("    reply=%r" % text[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
