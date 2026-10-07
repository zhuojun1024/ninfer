"""Build the fixed-tail / swept-history streams for the gold-standard depth sweep.

Design
------
TARGET BLOCK = the last BLOCK_TOKENS tokens of the source stream.  It is byte-identical
in every run, so the text being scored never changes.

Run k = the last (K + BLOCK_TOKENS) tokens of the source stream, i.e. the block preceded
by K tokens of its own natural context.  With --context >= stream length every run is a
single window, so the block's target t is predicted from [0, t) -- exactly t tokens of
history, all of it the block's own preceding text.

Consequences that make this the clean design:
  * the text of every scored token is identical across runs  -> content confound removed;
  * the block's immediate predecessor is the same text in every run with K >= 1
    -> no join/domain-shift artefact, the only variable is how far back the history goes.

Cuts are snapped to a line start; tokenisation of the block's tail is stable, and the
first few block tokens (which can absorb a boundary merge shift) are dropped in analysis.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
# Scratch stream built from the eval corpora; see profiles/kv-quantization/report.md.
SOURCE = os.path.join(REPO, "_temp", "20261008-0920_stream.txt")
OUT = os.path.join(REPO, "_temp", "scoreSweep2")

SOURCE_TOKENS = 257439
BLOCK_TOKENS = 1024

# nominal history lengths (tokens of context preceding the fixed block)
DEPTHS = [0, 256, 1024, 4096, 16384, 32768, 65536, 98304, 130048,
          163840, 196608, 229376, 256415]


def main():
    os.makedirs(OUT, exist_ok=True)
    text = open(SOURCE, encoding="utf-8").read()
    ratio = len(text) / float(SOURCE_TOKENS)
    print("source chars %d / %d tokens = %.4f chars per token" % (len(text), SOURCE_TOKENS, ratio))

    for depth in DEPTHS:
        # token index where the visible window must start
        start_token = SOURCE_TOKENS - BLOCK_TOKENS - depth
        offset = int(start_token * ratio)
        newline = text.find("\n", offset)
        offset = newline + 1 if newline >= 0 else offset
        window = text[offset:]
        path = os.path.join(OUT, "stream_k%06d.txt" % depth)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(window)
        print("  k=%-7d start_token=%-7d chars=%-8d file=%8d bytes"
              % (depth, start_token, offset, os.path.getsize(path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
