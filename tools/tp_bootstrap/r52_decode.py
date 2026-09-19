#!/usr/bin/env python3
"""Deterministic decode measurement at the shipped ceiling: per-request temperature 0, a fixed
prose prompt, usage-reported token counts, and the MTP round counters from the serve log."""
import json, re, subprocess, sys, time, urllib.request
URL = "http://127.0.0.1:8088/v1/chat/completions"
MODEL = "qwen3.8-27b"
LOG = "/home/zhuojun/prof/serve_supervised.log"

def rounds():
    out = subprocess.run(["bash", "-lc", "grep -a '\[mtp\] round' %s | tail -1" % LOG],
                         capture_output=True, text=True).stdout.strip()
    m = re.search(r"rate=(\d+)/(\d+)", out)
    tot = subprocess.run(["bash", "-lc", "grep -ac '\[mtp\] round' %s" % LOG],
                         capture_output=True, text=True).stdout.strip()
    return (int(m.group(1)), int(m.group(2)), int(tot or 0), out)

def post(payload, timeout=900):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode())
    return time.time() - t0, data

def bench(max_tokens, temp, label):
    a = rounds()
    total, data = post({"model": MODEL,
        "messages": [{"role": "user", "content":
            "Write a detailed essay about the history of open-sea navigation, at least 400 words. "
            "Do not stop before 400 words."}],
        "max_tokens": max_tokens, "temperature": temp, "stream": False})
    b = rounds()
    used = data.get("usage", {}).get("completion_tokens", 0)
    acc = (b[0] - a[0]) / max(b[1] - a[1], 1)
    tps = used / total
    print("%-22s tokens=%3d wall=%6.2fs -> %5.2f tok/s | rounds=%d drafts_ok=%d/%d=%.3f"
          % (label, used, total, tps, b[2] - a[2], b[0] - a[0], b[1] - a[1], acc))
    return tps

def prefill(label):
    prompt = ("The quick brown fox jumps over the lazy dog. " * 1600) + " Summarize."
    total, data = post({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 1, "temperature": 0, "stream": False})
    used = data.get("usage", {}).get("prompt_tokens", 0)
    print("%-22s prompt=%d prefill=%6.2fs -> %7.1f tok/s" % (label, used, total, used / total))

prefill("prefill ~16k")
bench(64, 0, "decode t=0 64tok")
bench(512, 0, "decode t=0 512tok")
