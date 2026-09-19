#pragma once

// Single authority for the per-device opt-in to more than 48 KiB of dynamic shared memory.
//
// cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize) is a per-device property of a
// function: it configures the current device's copy, and only that one. A function-local `static`
// guard therefore runs once per process and leaves every device except the first unconfigured, and
// the next >48 KiB launch from such a device fails with cudaErrorInvalidValue. A tensor-parallel
// run launches the same instantiation from every shard's context, so this must be keyed on the
// (kernel, device) pair.

#include "core/device.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

inline void ensure_max_dynamic_shared_memory(const void* kernel, int bytes) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    // The memo is a small fixed table: the engine launches a bounded number of distinct kernels,
    // each from at most one device per shard. Once full it degrades to an unconditional
    // cudaFuncSetAttribute, which stays correct.
    struct OptIn {
        const void* kernel;
        int device;
    };
    constexpr int kSlots = 64;
    static OptIn done[kSlots] = {};
    static int count           = 0;
    for (int i = 0; i < count; ++i) {
        if (done[i].kernel == kernel && done[i].device == device) { return; }
    }
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    if (count < kSlots) {
        done[count].kernel = kernel;
        done[count].device = device;
        ++count;
    }
}

} // namespace ninfer::ops::detail
