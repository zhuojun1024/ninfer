"""Interpret the current compressed-tensors FP8/NVFP4 fields and scale semantics.

The matrix resolver also accepts direct tensors. Resolution remains lazy so a
recipe can replace an unused checkpoint source before its weights are inspected.
"""

from __future__ import annotations

from math import prod
import struct

import torch

from tools.artifact.codecs.fp8_row import validate_fp8_row_words
from tools.artifact.formats import valid_positive_fp32_word
from .logical import EncodedRows, LogicalSource
from .safetensors import SafetensorsSource, tensor_source


def _nvfp4_packed_name(store: SafetensorsSource, prefix: str) -> str:
    """Name the packed NVFP4 codes for a supported export layout.

    compressed-tensors writes them as `weight_packed`; NVIDIA ModelOpt writes them
    as the plain `weight` tensor, so the name alone decides the layout.
    """
    if store.has(prefix + ".weight_packed"):
        return prefix + ".weight_packed"
    if store.has(prefix + ".weight_scale_2"):
        return prefix + ".weight"
    raise ValueError(f"{prefix}: no supported NVFP4 packed codes")


def _nvfp4_scale_name(store: SafetensorsSource, prefix: str) -> tuple[str, bool]:
    """Name the per-tensor NVFP4 weight scale and whether it must be inverted.

    compressed-tensors stores a divisor as `weight_global_scale`; NVIDIA ModelOpt
    stores the per-tensor multiplier as `weight_scale_2`. NInfer keeps a divisor,
    so the ModelOpt multiplier is inverted when it is read.
    """
    if store.has(prefix + ".weight_packed"):
        return prefix + ".weight_global_scale", False
    if store.has(prefix + ".weight_scale_2"):
        return prefix + ".weight_scale_2", True
    raise ValueError(f"{prefix}: no supported NVFP4 per-tensor weight scale")


def _nvfp4_input_name(
    store: SafetensorsSource, prefix: str
) -> tuple[str, bool] | None:
    """Name the NVFP4 activation scale and whether it must be inverted.

    compressed-tensors stores a divisor as `input_global_scale`; NVIDIA ModelOpt
    stores the per-tensor multiplier as `input_scale`. A missing tensor means the
    export was not calibrated for encoded activations.
    """
    if store.has(prefix + ".input_global_scale"):
        return prefix + ".input_global_scale", False
    if store.has(prefix + ".input_scale"):
        return prefix + ".input_scale", True
    return None


def _tensor_divisor(store: SafetensorsSource, name: str, inverted: bool) -> bytes:
    """Read one per-tensor scale and return the divisor NInfer stores."""
    word = _divisor_word(store, name)
    if not inverted:
        return word
    reciprocal = struct.pack("<f", 1.0 / struct.unpack("<f", word)[0])
    if not valid_positive_fp32_word(struct.unpack("<I", reciprocal)[0]):
        raise ValueError(f"{name}: reciprocal is not a positive finite FP32")
    return reciprocal


def _divisor_word(store: SafetensorsSource, name: str) -> bytes:
    info = store.describe(name)
    if info.dtype != "F32" or prod(info.shape) != 1:
        raise ValueError(f"{name}: expected a source FP32 scalar")
    tensor = store.read_flat(name)
    raw = tensor.view(torch.uint8).numpy().tobytes()
    if not valid_positive_fp32_word(struct.unpack("<I", raw)[0]):
        raise ValueError(f"{name}: divisor must be finite and positive")
    return raw


def compressed_matrix_source(
    store: SafetensorsSource, prefix: str, shape: tuple[int, int], format: str
) -> LogicalSource:
    """Interpret the current compressed-tensors NVFP4 or per-row FP8 representation."""
    if format not in ("nvfp4", "fp8_e4m3fn_row_bf16"):
        raise ValueError(f"unsupported encoded source format {format}")
    n, k = shape
    if format == "nvfp4" and k % 16:
        raise ValueError(f"{prefix}: NVFP4 source K must be divisible by 16")

    def signature(name: str, expected: tuple[int, ...], dtype: str) -> None:
        info = store.describe(name)
        if info.shape != expected or info.dtype != dtype:
            raise ValueError(
                f"{name}: expected {dtype}{expected}, got {info.dtype}{info.shape}"
            )

    def divisor(suffix: str) -> bytes:
        return _divisor_word(store, f"{prefix}.{suffix}")

    def tensor_divisor() -> bytes:
        return _tensor_divisor(store, *_nvfp4_scale_name(store, prefix))

    def tensor_input_divisor() -> bytes:
        named = _nvfp4_input_name(store, prefix)
        if named is None:
            raise ValueError(f"{prefix}: no supported NVFP4 activation scale")
        return _tensor_divisor(store, *named)

    def encoded(begin: int, end: int) -> EncodedRows:
        if not 0 <= begin < end <= n:
            raise ValueError(f"{prefix}: invalid encoded rows [{begin},{end})")
        if format == "nvfp4":
            packed, scale = _nvfp4_packed_name(store, prefix), f"{prefix}.weight_scale"
            signature(packed, (n, k // 2), "U8")
            signature(scale, (n, k // 16), "F8_E4M3")
            codes = store.read_flat(packed, begin * (k // 2), end * (k // 2)).reshape(
                end - begin, k // 2
            )
            scales = (
                store.read_flat(scale, begin * (k // 16), end * (k // 16))
                .view(torch.uint8)
                .reshape(end - begin, k // 16)
            )
            if bool((scales > 0x7E).any()):
                raise ValueError(f"{scale}: expected nonnegative finite E4M3FN scales")
            return EncodedRows(format, codes, scales, tensor_divisor())
        weight, scale = f"{prefix}.weight", f"{prefix}.weight_scale"
        signature(weight, shape, "F8_E4M3")
        info = store.describe(scale)
        codes = (
            store.read_flat(weight, begin * k, end * k)
            .view(torch.uint8)
            .reshape(end - begin, k)
        )
        if info.dtype == "BF16" and prod(info.shape) == n:
            scales = store.read_flat(scale, begin, end)
        elif info.dtype == "F32" and prod(info.shape) == 1:
            # ModelOpt scales a whole FP8 tensor with one FP32 factor. The stored row
            # representation repeats it, so every code word survives unchanged.
            factor = struct.unpack("<f", store.read_flat(scale).numpy().tobytes())[0]
            scales = torch.full((end - begin,), factor, dtype=torch.bfloat16)
        else:
            raise ValueError(f"{scale}: expected per-row BF16 or per-tensor F32 scales")
        validate_fp8_row_words(codes, scales)
        return EncodedRows(format, codes, scales)

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        first, last = begin // k, (end + k - 1) // k
        words = encoded(first, last)
        if format == "fp8_e4m3fn_row_bf16":
            values = (
                words.codes.view(torch.float8_e4m3fn).float()
                * words.scales.float()[:, None]
            )
        else:
            codes = torch.stack((words.codes & 15, words.codes >> 4), dim=-1).reshape(
                last - first, k
            )
            values_table = torch.tensor(
                [
                    0.0,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                    -0.0,
                    -0.5,
                    -1.0,
                    -1.5,
                    -2.0,
                    -3.0,
                    -4.0,
                    -6.0,
                ]
            )
            values = values_table[codes.long()]
            scales = (
                words.scales.view(torch.float8_e4m3fn)
                .float()
                .repeat_interleave(16, dim=1)
            )
            values = values * scales / struct.unpack("<f", words.weight_divisor)[0]
        return values.reshape(-1)[begin - first * k : end - first * k]

    return LogicalSource(
        shape,
        f"{store.path}:{prefix} ({format})",
        read,
        encoded,
        (tensor_divisor if format == "nvfp4" else None),
        (tensor_input_divisor if format == "nvfp4" else None),
    )


def matrix_source(
    store: SafetensorsSource,
    name: str,
    shape: tuple[int, int],
    format: str | None = None,
) -> LogicalSource:
    """Resolve the selected matrix's encoding lazily, after recipe source overrides."""
    prefix = name.removesuffix(".weight")
    resolved: LogicalSource | None = None

    def resolve() -> LogicalSource:
        nonlocal resolved
        if resolved is None:
            actual = format
            if actual is None and (
                store.has(prefix + ".weight_packed")
                or store.has(prefix + ".weight_scale_2")
            ):
                actual = "nvfp4"
            if actual is None and store.describe(name).dtype == "F8_E4M3":
                actual = "fp8_e4m3fn_row_bf16"
            resolved = (
                tensor_source(store, name, shape)
                if actual is None
                else compressed_matrix_source(store, prefix, shape, actual)
            )
        return resolved

    def encoded(begin: int, end: int) -> EncodedRows:
        reader = resolve().read_encoded
        if reader is None:
            raise ValueError(f"{name}: selected source does not provide encoded rows")
        return reader(begin, end)

    def divisor(which: str) -> bytes:
        read = getattr(resolve(), which)
        if read is None:
            raise ValueError(f"{name}: selected source does not provide {which}")
        return read()

    return LogicalSource(
        shape,
        f"{store.path}:{name}",
        lambda begin, end: resolve().values(begin, end),
        encoded,
        lambda: divisor("weight_divisor"),
        lambda: divisor("input_divisor"),
    )
