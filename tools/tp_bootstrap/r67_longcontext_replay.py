"""Replay a growing coding-agent conversation against a live ninfer-serve.

The 2026-09-24 server crash (PLAN.md section 8 worklog) happened on the fifth request of one
growing conversation, right after the first prompt passed ~13k tokens with ~99% prefix reuse, and
surfaced as cudaErrorIllegalAddress at the DFlash2 round's licensed-token copy. This probe rebuilds
that shape: a long synthetic repository dump, then small turns that only append to the tail so every
request after the first is a cached prefix plus a re-walked suffix.

  python tools/tp_bootstrap/r67_longcontext_replay.py --turn-tokens 12000 --turns 5

It never starts or stops an engine; point it at a server the caller owns.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

PARAGRAPH = (
    "def materialize_rows(store, prefix, begin, end, group_size):\n"
    "    # Decode the packed group codes with their stored scales; never re-derive the scale.\n"
    "    codes = store.read_flat(prefix + '.weight', begin * 5120, end * 5120)\n"
    "    scales = store.read_flat(prefix + '.weight_scale', begin * 160, end * 160)\n"
    "    return codes, scales, group_size, prefix, begin, end\n"
)
SENTENCE = (
    "The engine streams the shard-local delta through the write-order chain and the peer applies it "
    "before the next collective, so the rendezvous id has to be authoritative rather than inferred. "
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


def build_block(words: int, nonce: str) -> str:
    # Roughly 1.3 tokens per word for this tokenizer; the caller verifies with usage.prompt_tokens.
    body = []
    while sum(len(part.split()) for part in body) < words:
        body.append(PARAGRAPH if len(body) % 2 == 0 else SENTENCE)
    return "Repository dump " + nonce + ":\n" + "".join(body)


def post(url: str, payload: dict, timeout: int = 900):
    body = json.dumps(payload).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        parsed = json.loads(response.read())
    return parsed, time.time() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:3456/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--turn-tokens", type=int, default=12000)
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.8)
    args = parser.parse_args()

    url = args.base_url.rstrip("/") + "/chat/completions"
    system = {
        "role": "system",
        "content": "You are a coding agent working in a large repository. Use the tools when asked.",
    }
    messages = [system]
    nonce = time.strftime("%H%M%S")
    words = int(args.turn_tokens / 1.3)
    messages.append({"role": "user", "content": build_block(words, nonce) + "\nSummarize the dump."})

    for turn in range(1, args.turns + 1):
        payload = {
            "model": args.model,
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "stream": False,
            "tools": TOOLS,
        }
        try:
            parsed, elapsed = post(url, payload)
        except urllib.error.HTTPError as error:
            print(f"turn {turn}: HTTP {error.code} {error.read()[:200]!r}", flush=True)
            return 1
        except Exception as error:  # connection death is the signal we are looking for
            print(f"turn {turn}: TRANSPORT FAILURE {type(error).__name__}: {error}", flush=True)
            return 2
        usage = parsed.get("usage", {})
        choice = parsed["choices"][0]
        message = choice["message"]
        text = (message.get("reasoning_content") or "") + (message.get("content") or "")
        calls = message.get("tool_calls") or []
        print(
            "turn {0}: prompt={1} output={2} finish={3} wall={4:.2f}s tool_calls={5}".format(
                turn, usage.get("prompt_tokens"), usage.get("completion_tokens"),
                choice.get("finish_reason"), elapsed, len(calls),
            ),
            flush=True,
        )
        if calls:
            messages.append({"role": "assistant", "content": message.get("content") or "",
                             "tool_calls": calls})
            messages.append({"role": "tool", "tool_call_id": calls[0]["id"],
                             "content": "materialize_rows(): decoded 128 rows, no errors."})
        else:
            messages.append({"role": "assistant", "content": message.get("content") or text})
        messages.append({
            "role": "user",
            "content": "Now check the rendezvous code path and report the next step. Nonce " + str(turn),
        })
    print("REPLAY_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
