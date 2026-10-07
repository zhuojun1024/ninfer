"""Build the fixed-block / swept-history streams for the gold-standard depth sweep.

Design
------
Each stream is  [ filler prefix of ~d tokens ] + BLOCK.  BLOCK is one fixed English
passage taken from wikitext (a different document than the pg19 filler, so the model
cannot have memorised it from the history).  With --context >= len(stream) every run
is a single window, so the BLOCK's targets are predicted from [0, t) where t is the
target index -- i.e. from exactly d + (offset inside the block) tokens of history.

Only the LAST SCORED_TOKENS of the block are analysed: the first part of the block
absorbs the join artefact, so the analysed tokens have their entire local context
inside the block and the only thing that changes across runs is the far history.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
# Scratch stream built from the eval corpora; see profiles/kv-quantization/report.md.
FILLER = os.path.join(REPO, "_temp", "20261008-0920_stream.txt")
WIKITEXT = os.path.join(REPO, "eval", "corpora", "perplexity-1m", "data", "wikitext", "00.txt")
OUT = os.path.join(REPO, "_temp", "scoreSweep")

FILLER_TOKENS = 257439          # measured earlier for this exact file
DEPTHS = [0, 1024, 4096, 16384, 32768, 65536, 98304, 131072,
          163840, 196608, 229376, 257439]


def build_block():
    """One self-contained English passage, long enough that the scored tail is deep inside."""
    text = open(WIKITEXT, encoding="utf-8", errors="replace").read()
    start = text.find("\n\n", 4000)
    if start < 0:
        start = 4000
    start += 2
    end = text.find("\n\n", start + 5200)
    if end < 0:
        end = start + 5200
    block = text[start:end].strip("\n")
    return block


def main():
    os.makedirs(OUT, exist_ok=True)
    filler = open(FILLER, encoding="utf-8").read()
    block = build_block()
    print("filler chars %d (assumed %d tokens)" % (len(filler), FILLER_TOKENS))
    print("block chars %d (roughly %d tokens)" % (len(block), len(block) // 4.0))

    ratio = len(filler) / float(FILLER_TOKENS)
    written = []
    for depth in DEPTHS:
        offset = int(depth * ratio)
        # snap forward to the next line start so we never cut a word in half
        newline = filler.find("\n", offset)
        offset = newline + 1 if newline >= 0 else offset
        stream = filler[:offset] + "\n\n" + block + "\n"
        path = os.path.join(OUT, "stream_d%06d.txt" % depth)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(stream)
        written.append((depth, offset, os.path.getsize(path)))
    for depth, offset, size in written:
        print("  d=%-7d filler chars=%-8d file=%8d bytes" % (depth, offset, size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
