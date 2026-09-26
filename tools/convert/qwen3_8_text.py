"""Qwen3.8-27B dense text tower geometry and llama.cpp's Qwen3.5 exporter conventions.

A GGUF recipe for this tower reads rows and small tensors straight out of a llama.cpp export, which
does not store the logical parameters as they are computed: value heads are tiled, zero-centred norms
keep a stored `1 + w`, and a query/output-gate pair is head-interleaved. This module owns the tower's
fixed geometry and the row maps that undo those conventions, so a recipe only names the tensors it
reads.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from .sources.gguf import GGUFFile
from .sources.logical import LogicalSource, array_source

HIDDEN = 5120
INTERMEDIATE = 17408
VOCABULARY = 248320
LAYERS = 64
ATTENTION_HEADS = 24
ATTENTION_HEAD_DIM = 256
ATTENTION_QUERY_ROWS = ATTENTION_HEADS * ATTENTION_HEAD_DIM
ATTENTION_KV_ROWS = 1024
GDN_KEY_HEADS = 16
GDN_VALUE_HEADS = 48
GDN_HEAD_DIM = 128
GDN_KEY_DIM = GDN_KEY_HEADS * GDN_HEAD_DIM
GDN_VALUE_DIM = GDN_VALUE_HEADS * GDN_HEAD_DIM
GDN_CHANNELS = 2 * GDN_KEY_DIM + GDN_VALUE_DIM
GDN_TAPS = 4


def full_attention(layer: int) -> bool:
    return layer % 4 == 3


def tiled_to_grouped_permutation() -> np.ndarray:
    """For grouped value head `h = k * 3 + r`, the exporter's tiled position `r * 16 + k`.

    llama.cpp's exporter stores value heads tiled so that a plain repeat broadcasts the key
    heads; NInfer pairs value head `h` with key head `h // 3` and wants the grouped order.
    """

    heads = np.arange(GDN_VALUE_HEADS)
    per_key = GDN_VALUE_HEADS // GDN_KEY_HEADS
    return (heads % per_key) * GDN_KEY_HEADS + heads // per_key


def untile(array: np.ndarray, head_dim: int) -> np.ndarray:
    """Reorder axis 0 from the tiled value-head order back to grouped."""

    if array.shape[0] != GDN_VALUE_HEADS * head_dim:
        raise ValueError(
            f"axis 0 has {array.shape[0]} entries, expected {GDN_VALUE_HEADS * head_dim}"
        )
    view = array.reshape(GDN_VALUE_HEADS, head_dim, *array.shape[1:])
    return np.ascontiguousarray(view[tiled_to_grouped_permutation()].reshape(array.shape))


RowMap = Callable[[int, int], np.ndarray]


def rows(base: int = 0) -> RowMap:
    return lambda begin, end: np.arange(begin, end, dtype=np.int64) + base


def attention_rows(gate: bool) -> RowMap:
    """The query (or output gate) rows of the head-interleaved `attn_q`."""

    d = ATTENTION_HEAD_DIM

    def select(begin: int, end: int) -> np.ndarray:
        r = np.arange(begin, end, dtype=np.int64)
        return (r // d) * 2 * d + (d if gate else 0) + r % d

    return select


def untiled_rows(base: int) -> RowMap:
    """Grouped value-head rows of a tiled family starting at source row `base`."""

    permutation = tiled_to_grouped_permutation()

    def select(begin: int, end: int) -> np.ndarray:
        r = np.arange(begin, end, dtype=np.int64)
        return base + permutation[r // GDN_HEAD_DIM] * GDN_HEAD_DIM + r % GDN_HEAD_DIM

    return select


def flat_rows(read_rows, k: int):
    """Address a row source as a flat range, reading only the rows the slice covers."""

    def read(begin: int, end: int) -> torch.Tensor:
        first, last = begin // k, -(-end // k)
        values = read_rows(first, last).reshape(-1)
        return values[begin - first * k : end - first * k]

    return read


def direct_source(values: np.ndarray, dtype: torch.dtype, label: str) -> LogicalSource:
    return array_source(torch.from_numpy(np.ascontiguousarray(values)).to(dtype), label)


def norm_source(gguf: GGUFFile, tensor: str, offset: bool) -> LogicalSource:
    """A norm's BF16 weights, dropping the `1 + w` the exporter stores when `offset`."""

    words = gguf.read_direct(tensor).astype(np.float32)
    return direct_source(words - 1.0 if offset else words, torch.bfloat16, tensor)
