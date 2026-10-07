"""Harder variant of the prefix-chain probe.

The 8-needle version saturates (8/8 at every length up to 250K on both arms),
so it cannot see a difference between KV formats. This variant raises the
difficulty in two ways at zero extra prefill cost, because everything it adds
lives in the preamble that every request in the chain shares:

  * 32 needles instead of 8      -> 32 independent retrievals, score is x/32
  * one 39-char random access token that must be copied verbatim
                                 -> character-level score, not just argmax

It reuses the chain sweep machinery unchanged (same prefix-chain structure,
same output schema, same CLI).

Usage:
  python profiles/kv-quantization/tools/niah_hard_chain.py --arm int8 --port 18080 \
      --out profiles/kv-quantization/data/niah_chain_hard_int8.json
"""

from __future__ import annotations

import importlib.util
import pathlib
import random

BASE = str(pathlib.Path(__file__).with_name("niah_chain_sweep.py"))

spec = importlib.util.spec_from_file_location("chain", BASE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

CITIES = (
    "Amsterdam", "Athens", "Auckland", "Bangkok", "Barcelona", "Berlin", "Bogota",
    "Brisbane", "Brussels", "Budapest", "Cairo", "Calgary", "Chicago", "Copenhagen",
    "Dakar", "Dublin", "Edinburgh", "Florence", "Geneva", "Helsinki", "Istanbul",
    "Jakarta", "Kyoto", "Lagos", "Lisbon", "Madrid", "Melbourne", "Montreal",
    "Nairobi", "Oslo", "Prague", "Reykjavik",
)
rng = random.Random(20261008)
m.NEEDLES = tuple((city, f"{rng.randint(10000, 99999)}") for city in CITIES)
m.ACCESS = "XK4Q9M2B7T1V8W3N6R5P2H9D4C8L1G7F"

m.PREAMBLE = (
    "Reference sheet. Each line records the vault code of one office, followed by "
    "one access token.\n"
    + "".join(f"The vault code for the {city} office is {code}.\n" for city, code in m.NEEDLES)
    + f"\nAccess token: {m.ACCESS}\n\n"
)
m.QUESTION = (
    "\n\nQuestion: The reference sheet at the very beginning of this document lists the "
    "vault code of 32 offices and one access token. For each of those 32 offices, state its "
    "vault code, one line per office in the form `City = Code`. Then, on the final line, "
    "output the access token exactly as written. Output nothing else."
)

if __name__ == "__main__":
    raise SystemExit(m.main())
