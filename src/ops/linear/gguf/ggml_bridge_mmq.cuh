#pragma once

// Host side of ggml's integer tensor-core matrix kernel (mmq.cuh launch_mul_mat_q), without ggml's
// backend context: the caller owns the activation, the FP32 output and the stream-k fixup plane.

#include "ggml_bridge_internal.cuh"

#include <climits>

namespace ninfer::ops::gguf::detail {

struct MatrixLaunch {
    int J                   = 0;
    int I                   = 0;
    int threads             = 0;
    std::size_t shared      = 0;
    int row_tiles           = 0;
    int column_tiles        = 0;
    bool stream_k           = false;
    int blocks              = 0;
    bool fixup              = false;
};

template <ggml_type type>
MatrixLaunch select_matrix_launch(int rows, int columns, int max_J = INT_MAX) {
    const DeviceFacts& device = device_facts();
    MatrixLaunch best;
    int best_tiles = INT_MAX;
    for (const int J : kMatrixColumnTiles) {
        if (J > max_J) { continue; }
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, false, device.cc);
        if (config.type == GGML_TYPE_COUNT) { continue; }
        const std::size_t shared = mmq_get_nbytes_shared(config, device.cc);
        if (shared > device.shared_per_block_optin) { continue; }
        const int tiles = (columns + J - 1) / J;
        if (tiles < best_tiles) {
            best_tiles        = tiles;
            best.J            = J;
            best.I            = config.I;
            best.threads      = config.nthreads;
            best.shared       = shared;
            best.stream_k     = config.stream_k;
        }
        if (tiles == 1) { break; }
    }
    if (best.J == 0) { throw std::invalid_argument("gguf matrix product: no tile fits this device"); }
    best.row_tiles    = (rows + best.I - 1) / best.I;
    best.column_tiles = (columns + best.J - 1) / best.J;
    const int tiles   = best.row_tiles * best.column_tiles;
    if (best.stream_k) {
        const int waves      = (tiles + device.sm_count - 1) / device.sm_count;
        const int efficiency = 100 * tiles / (device.sm_count * waves);
        best.blocks          = efficiency >= 90 ? tiles : device.sm_count;
        best.fixup           = tiles % best.blocks != 0;
    } else {
        best.blocks = tiles;
    }
    return best;
}

template <ggml_type type>
std::size_t matrix_fixup_bytes_impl(int rows, int columns) {
    // The caller sizes one workspace for this product, and a refused tile falls back to a narrower
    // one, so the reservation has to cover every candidate rather than the widest alone.
    std::size_t bytes = 0;
    for (int max_J = INT_MAX;;) {
        const MatrixLaunch launch = select_matrix_launch<type>(rows, columns, max_J);
        if (launch.fixup) {
            const std::size_t candidate =
                std::size_t(launch.blocks) * launch.J * launch.I * sizeof(float);
            if (candidate > bytes) { bytes = candidate; }
        }
        if (launch.J <= kMatrixColumnTiles[0]) { break; }
        max_J = launch.J - 1;
    }
    return bytes;
}

// A rejected launch is a host-side configuration error, so the message carries the geometry that
// produced it: without it the CUDA error string alone says nothing about which product failed.
// The kernel's own attributes say whether the dynamic shared memory limit was actually raised.
template <ggml_type type, int J>
[[noreturn]] inline void throw_launch_error(cudaError_t status, const MatrixLaunch& launch,
                                            int rows, int k, int columns) {
    cudaFuncAttributes attrs{};
    const cudaError_t attr_status = cudaFuncGetAttributes(&attrs, mul_mat_q<type, J, false>);
    int device         = 0;
    int default_shared = 0;
    (void)cudaGetDevice(&device);
    (void)cudaDeviceGetAttribute(&default_shared, cudaDevAttrMaxSharedMemoryPerBlock, device);
    throw std::runtime_error(
        "gguf matrix product launch: " + std::string(cudaGetErrorString(status)) +
        " (rows=" + std::to_string(rows) + ", k=" + std::to_string(k) +
        ", columns=" + std::to_string(columns) + ", J=" + std::to_string(launch.J) +
        ", I=" + std::to_string(launch.I) + ", threads=" + std::to_string(launch.threads) +
        ", shared=" + std::to_string(launch.shared) +
        ", row_tiles=" + std::to_string(launch.row_tiles) +
        ", column_tiles=" + std::to_string(launch.column_tiles) +
        ", blocks=" + std::to_string(launch.blocks) +
        ", stream_k=" + (launch.stream_k ? "true" : "false") +
        ", fixup=" + (launch.fixup ? "true" : "false") +
        ", kernel_static_shared=" + std::to_string(attrs.sharedSizeBytes) +
        ", kernel_max_dynamic_shared=" + std::to_string(attrs.maxDynamicSharedSizeBytes) +
        ", kernel_max_threads=" + std::to_string(attrs.maxThreadsPerBlock) +
        ", kernel_regs=" + std::to_string(attrs.numRegs) +
        ", kernel_local=" + std::to_string(attrs.localSizeBytes) +
        ", kernel_ptx=" + std::to_string(attrs.ptxVersion) +
        ", kernel_binary=" + std::to_string(attrs.binaryVersion) +
        ", kernel_cluster_required=" + std::to_string(attrs.clusterDimMustBeSet) +
        ", device_default_block_shared=" + std::to_string(default_shared) +
        ", kernel_attrs=" + std::string(cudaGetErrorString(attr_status)) +
        ", device_optin_shared=" + std::to_string(device_facts().shared_per_block_optin) +
        ", device_sm_count=" + std::to_string(device_facts().sm_count) + ")");
}

// Dynamic shared memory above the architectural default needs a per-function, per-device opt-in.
// The same instantiation is launched from every shard's context, so the memo is keyed on the
// device rather than kept function-locally.
template <ggml_type type, int J>
void ensure_matrix_shared_memory(std::size_t shared) {
    if (shared <= kMatrixDefaultSharedBytes) { return; }
    int device = 0;
    check(cudaGetDevice(&device), "device");
    static bool raised[16] = {};
    if (raised[device & 15]) { return; }
    check(cudaFuncSetAttribute(mul_mat_q<type, J, false>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(device_facts().shared_per_block_optin)),
          "matrix shared memory");
    raised[device & 15] = true;
}

// Returns cudaSuccess, or the status of the rejected launch so that the caller can try a narrower
// tile. A rejection is host-side and executes nothing, so retrying with another tile is safe.
template <ggml_type type, int J>
cudaError_t launch_matrix(const MatrixLaunch& launch, const char* weight, int stride_row_blocks,
                          int rows, int k, const int* activation, int columns, float* out,
                          std::int64_t out_column_stride, float* fixup, cudaStream_t stream) {
    if (rows <= 0 || k <= 0 || columns <= 0 || launch.row_tiles <= 0 || launch.column_tiles <= 0 ||
        launch.blocks <= 0 || launch.threads % WARP_SIZE != 0 || launch.I % WARP_SIZE != 0) {
        throw std::invalid_argument(
            "gguf matrix product: degenerate launch geometry (rows=" + std::to_string(rows) +
            ", k=" + std::to_string(k) + ", columns=" + std::to_string(columns) + ")");
    }
    if (launch.fixup && fixup == nullptr) {
        throw std::invalid_argument("gguf matrix product: this launch needs a fixup plane");
    }
    ensure_matrix_shared_memory<type, J>(launch.shared);
    constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    const uint3 blocks_fd   = init_fastdiv_values(k / qk);
    const uint3 one_fd      = init_fastdiv_values(1);
    const uint3 ntx_fd      = init_fastdiv_values(launch.column_tiles);
    const dim3 block_dims(WARP_SIZE, launch.threads / WARP_SIZE, 1);
    const int stride        = static_cast<int>(out_column_stride);
    const dim3 grid         = launch.stream_k ? dim3(launch.blocks, 1, 1)
                                             : dim3(launch.row_tiles, launch.column_tiles, 1);
    const auto fire         = [&] {
        mul_mat_q<type, J, false><<<grid, block_dims, launch.shared, stream>>>(
            weight, activation, nullptr, nullptr, out, launch.fixup ? fixup : nullptr, nullptr,
            blocks_fd, rows, columns, stride_row_blocks, columns, stride, one_fd, one_fd, 0, 0, 0,
            one_fd, one_fd, 0, 0, 0, ntx_fd);
        return cudaGetLastError();
    };
    const cudaError_t status = fire();
    if (status != cudaSuccess) { return status; }
    if (!launch.fixup) { return cudaSuccess; }
    const dim3 fixup_blocks(launch.blocks, launch.I / WARP_SIZE, 1);
    const dim3 fixup_dims(block_dims.x, block_dims.y / 2, 1);
    mul_mat_q_stream_k_fixup<type, J, false><<<fixup_blocks, fixup_dims, 0, stream>>>(
        nullptr, nullptr, out, fixup, blocks_fd, rows, columns, stride, one_fd, 0, one_fd, 0,
        ntx_fd);
    check(cudaGetLastError(), "matrix fixup launch");
    return cudaSuccess;
}

template <ggml_type type>
void matrix_product_impl(const void* weight, std::int64_t row_bytes, int rows, int k,
                         const void* activation, int columns, float* out,
                         std::int64_t out_column_stride, void* fixup, cudaStream_t stream) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int block_bytes = ggml_cuda_type_traits<type>::bs;
    if (k % 256 != 0 || rows % 128 != 0 || row_bytes % block_bytes != 0 ||
        row_bytes < std::int64_t(k / qk) * block_bytes) {
        throw std::invalid_argument("gguf matrix product: unsupported geometry");
    }
    const auto* x        = static_cast<const char*>(weight);
    const auto* y        = static_cast<const int*>(activation);
    auto* f              = static_cast<float*>(fixup);
    const int stride_row = static_cast<int>(row_bytes / block_bytes);
    // The widest tile needs more dynamic shared memory than the architectural default, and this
    // device can refuse that launch even after the kernel's own limit was raised for it. Nothing
    // ran, so a narrower tile is tried instead; only when every tile is refused is the failure
    // reported, with the geometry of the widest attempt.
    for (int max_J = INT_MAX;;) {
        const MatrixLaunch launch = select_matrix_launch<type>(rows, columns, max_J);
        cudaError_t status        = cudaSuccess;
        switch (launch.J) {
        case 16:
            status = launch_matrix<type, 16>(launch, x, stride_row, rows, k, y, columns, out,
                                             out_column_stride, f, stream);
            break;
        case 32:
            status = launch_matrix<type, 32>(launch, x, stride_row, rows, k, y, columns, out,
                                             out_column_stride, f, stream);
            break;
        case 64:
            status = launch_matrix<type, 64>(launch, x, stride_row, rows, k, y, columns, out,
                                             out_column_stride, f, stream);
            break;
        case 128:
            status = launch_matrix<type, 128>(launch, x, stride_row, rows, k, y, columns, out,
                                              out_column_stride, f, stream);
            break;
        default:
            throw std::invalid_argument("gguf matrix product: unexpected column tile");
        }
        if (status == cudaSuccess) { return; }
        if (status != cudaErrorInvalidValue || launch.J <= kMatrixColumnTiles[0]) {
            switch (launch.J) {
            case 16: throw_launch_error<type, 16>(status, launch, rows, k, columns);
            case 32: throw_launch_error<type, 32>(status, launch, rows, k, columns);
            case 64: throw_launch_error<type, 64>(status, launch, rows, k, columns);
            default: throw_launch_error<type, 128>(status, launch, rows, k, columns);
            }
        }
        max_J = launch.J - 1;
    }
}

} // namespace ninfer::ops::gguf::detail

#define NINFER_GGUF_MATRIX_INSTANCE(TYPE)                                                          \
    template std::size_t ninfer::ops::gguf::detail::matrix_fixup_bytes_impl<TYPE>(int, int);      \
    template void ninfer::ops::gguf::detail::matrix_product_impl<TYPE>(                           \
        const void*, std::int64_t, int, int, const void*, int, float*, std::int64_t, void*,        \
        cudaStream_t)
