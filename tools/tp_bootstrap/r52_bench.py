#!/usr/bin/env python3
"""Round 52 end-to-end benchmark at the shipped context ceiling.

Measures, against one running ninfer-serve:
  * decode throughput with MTP (streaming, tokens 2..N over the streaming window)
  * prefill throughput (time to first token on a fresh long prompt)
  * prefix reuse (same long prompt again)
  * a deep-context request at ~1/4 of the ceiling
  * concurrency C=2
Prints one summary line per measurement.
"""
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:8088/v1/chat/completions"
MODEL = "qwen3.8-27b"


def post(payload, stream=True):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    started = time.time()
    first = None
    chunks = 0
    text = []
    with urllib.request.urlopen(req, timeout=1800) as resp:
        if not stream:
            data = json.loads(resp.read().decode())
            total = time.time() - started
            txt = data["choices"][0]["message"]["content"]
            return {"ttfb": total, "total": total, "tokens": len(txt), "text": txt, "chunks": 1}
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload_line = line[5:].strip()
            if payload_line == "[DONE]":
                break
            if first is None:
                first = time.time()
            chunks += 1
            try:
                obj = json.loads(payload_line)
            except json.JSONDecodeError:
                continue
            delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
            if delta:
                text.append(delta)
    total = time.time() - started
    return {"ttfb": (first or total) - started, "total": total, "chunks": chunks,
            "text": "".join(text)}


def sentence(reps):
    return ("The quick brown fox jumps over the lazy dog. " * reps)


def main():
    out = {}
    warm = post({"model": MODEL, "messages": [{"role": "user", "content": "Say hi."}],
                 "max_tokens": 8, "stream": True}, stream=True)
    print("warmup ttfb=%.2fs total=%.2fs" % (warm["ttfb"], warm["total"]))

    # Decode: 512 tokens with MTP.
    dec = post({"model": MODEL,
                "messages": [{"role": "user", "content": "Write a long essay about the sea."}],
                "max_tokens": 512, "stream": True}, stream=True)
    decode_tps = (dec["chunks"] - 1) / max(dec["total"] - dec["ttfb"], 1e-6)
    print("decode mtp512 ttfb=%.2fs total=%.2fs chunks=%d decode=%.2f tok/s"
          % (dec["ttfb"], dec["total"], dec["chunks"], decode_tps))
    out["decode_tps"] = decode_tps

    # Prefill + reuse on a ~16k-token prompt.
    long_prompt = sentence(1600) + " Summarize the above in one short sentence."
    p1 = post({"model": MODEL, "messages": [{"role": "user", "content": long_prompt}],
               "max_tokens": 16, "stream": True}, stream=True)
    print("prefill ~16k fresh ttfb=%.2fs total=%.2fs" % (p1["ttfb"], p1["total"]))
    p2 = post({"model": MODEL, "messages": [{"role": "user", "content": long_prompt}],
               "max_tokens": 16, "stream": True}, stream=True)
    print("prefill ~16k repeated ttfb=%.2fs total=%.2fs" % (p2["ttfb"], p2["total"]))
    out["reuse_ttfb"] = p2["ttfb"]

    # Deep context: ~65k tokens, first token latency, then reuse.
    deep_prompt = sentence(6500) + " Summarize the above in one short sentence."
    d1 = post({"model": MODEL, "messages": [{"role": "user", "content": deep_prompt}],
               "max_tokens": 16, "stream": True}, stream=True)
    print("deep ~65k fresh ttfb=%.2fs total=%.2fs" % (d1["ttfb"], d1["total"]))
    d2 = post({"model": MODEL, "messages": [{"role": "user", "content": deep_prompt}],
               "max_tokens": 16, "stream": True}, stream=True)
    print("deep ~65k repeated ttfb=%.2fs total=%.2fs" % (d2["ttfb"], d2["total"]))
    out["deep_ttfb"] = d1["ttfb"]

    # Concurrency C=2.
    def one(i):
        return post({"model": MODEL,
                     "messages": [{"role": "user", "content": "Count from %d to %d." % (i, i + 20)}],
                     "max_tokens": 64, "stream": True}, stream=True)
    started = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        res = list(pool.map(one, (1, 2)))
    print("concurrency c=2 wall=%.2fs chunks=%s" % (time.time() - started,
                                                    [r["chunks"] for r in res]))
    out["c2"] = [r["chunks"] for r in res]
    print("SUMMARY " + json.dumps({k: round(v, 3) if isinstance(v, float) else v
                                  for k, v in out.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
