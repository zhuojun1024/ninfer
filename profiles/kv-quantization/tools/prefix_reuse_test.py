"""Does ninfer-serve reuse the KV of a strict prefix of a new request?

Sends three chat requests with a tiny max_tokens so the wall clock is dominated
by prefill:

  A  ~128K tokens                (cold)
  A  the same prompt again       (exact prefix -> should be decode only)
  B  A plus ~32K more tokens     (strict extension -> should be increment only)

If B comes back in roughly the time of a 32K prefill rather than a 160K one,
prefix reuse covers strict extensions and a length sweep can be run as one
prefix chain for the price of a single full prefill.

Usage:  python profiles/kv-quantization/tools/prefix_reuse_test.py <long-stream.txt>

The stream is any long UTF-8 text file; the test needs roughly 160K tokens of it.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
import urllib.request

if len(sys.argv) < 2:
    raise SystemExit(
        f"usage: {pathlib.Path(__file__).name} <long-stream.txt>  "
        "(any long UTF-8 text; about 160K tokens are consumed)"
    )
STREAM = pathlib.Path(sys.argv[1])
URL = "http://127.0.0.1:18080/v1/chat/completions"
MODEL = "qwen38-tp2"


def cut_at_newline(target_bytes: int) -> str:
    text = STREAM.read_text(encoding="utf-8")
    end = text.rfind("\n", 0, target_bytes)
    return text[: end if end > 0 else target_bytes]


def ask(prompt: str, label: str) -> float:
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 4,
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            payload = json.loads(resp.read())
        usage = payload.get("usage", {})
        elapsed = time.time() - started
        print(f"{label:<28} {elapsed:>8.1f}s  in={usage.get('prompt_tokens')} out={usage.get('completion_tokens')}",
              flush=True)
        return elapsed
    except Exception as exc:  # noqa: BLE001 - report and keep going
        elapsed = time.time() - started
        print(f"{label:<28} {elapsed:>8.1f}s  FAILED: {exc}", flush=True)
        return elapsed


def main() -> int:
    print(f"stream {STREAM} ({STREAM.stat().st_size} bytes)")
    short = cut_at_newline(520_000)   # ~128K tokens at ~4 bytes/token
    long_ = cut_at_newline(650_000)   # ~160K tokens
    print(f"prompt A {len(short)} chars   prompt B {len(long_)} chars\n", flush=True)

    a_cold = ask(short, "A ~128K cold")
    a_warm = ask(short, "A ~128K warm repeat")
    b_ext = ask(long_, "B ~160K strict extension")

    print()
    print(f"exact-prefix speedup : {a_cold / max(a_warm, 0.1):.1f}x  ({a_cold:.0f}s -> {a_warm:.0f}s)")
    if b_ext < 0.5 * a_cold:
        print(f"VERDICT: strict extensions are reused ({b_ext:.0f}s vs {a_cold:.0f}s cold) "
              "-> a prefix-chain sweep costs about one full prefill")
    else:
        print(f"VERDICT: NO reuse for strict extensions ({b_ext:.0f}s vs {a_cold:.0f}s cold) "
              "-> every length step pays a full prefill")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
