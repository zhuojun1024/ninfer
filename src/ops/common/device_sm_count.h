#pragma once

// Single authority for "how many streaming multiprocessors does the device I am launching on have".
//
// Several launchers size a grid, a prefetch cutoff or a resident-CTA budget in whole waves of the
// device. Those constants were swept on the 170-SM part they were measured on and then written as
// literals; on any other part -- a 36-SM RTX 5060 Ti, a 142-SM RTX 4090 -- the wave arithmetic they
// feed is wrong, because "170 blocks" is one wave there and about five elsewhere. Query the active
// device once per device and keep the measured 170 as the fallback for a failed query.
//
// The probe is per (device) rather than per process: a tensor-parallel run launches the same
// instantiation from every shard's context, and cudaGetDevice reports whichever one is active.

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

inline int device_sm_count(int fallback = 170) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        (void)cudaGetLastError();
        return fallback;
    }
    struct Probe {
        int device;
        int sm_count;
    };

    constexpr int kSlots = 8;
    static Probe probes[kSlots] = {};
    static int count            = 0;
    for (int i = 0; i < count; ++i) {
        if (probes[i].device == device) { return probes[i].sm_count; }
    }

    int sm_count = 0;
    if (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
        sm_count <= 0) {
        (void)cudaGetLastError();
        return fallback;
    }
    if (count < kSlots) {
        probes[count].device   = device;
        probes[count].sm_count = sm_count;
        ++count;
    }
    return sm_count;
}

} // namespace ninfer::ops::detail
