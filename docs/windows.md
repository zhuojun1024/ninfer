# Native Windows build and run

NInfer builds and runs natively on Windows with MSVC and the CUDA toolkit: the engine, the CLI and the
HTTP server are the same sources as the Linux build, with platform branches confined to file I/O, host
process queries, media acquisition, and one CUDA kernel-parameter detail.

Verified configuration: Visual Studio 2022 (MSVC 19.44) + CUDA 13.3 + Ninja, two RTX 5060 Ti 16 GiB
cards in a TP-2 pair (`--devices 0,1`), artifact `qwen3_8_27b_nvfp4.ninfer` (23.7 GB).

## Prerequisites

- Visual Studio 2022 with the C++ desktop workload (`vcvars64.bat`).
- CUDA toolkit 13.3 (`nvcc`); CMake and Ninja on `PATH`.
- An FFmpeg **development** prefix (headers plus import libraries) for media decoding. The shared
  BtbN build works; extract it anywhere and pass the prefix:
  `-DNINFER_FFMPEG_ROOT=D:/ffmpeg-dev/expanded/ffmpeg-master-latest-win64-gpl-shared`.
- A libcurl prefix for remote media acquisition (`tools/win_port/fetch_curl.ps1` downloads one). The
  import library is used, not the static archive: MSVC must link `libcurl.dll.a` and load
  `libcurl-x64.dll` at run time. Only the CLI/server targets need it.

## Build

```powershell
pwsh -File tools/win_port/build.ps1 -Configure        # configure and build (Release, sm_120a)
pwsh -File tools/win_port/build.ps1                  # incremental build
pwsh -File tools/win_port/build.ps1 -Configure -Apps 0   # core libraries only
```

Products land in `build-win/apps/{ninfer,ninfer-serve,ninfer-perplexity}.exe`. `build.ps1` imports the
`vcvars64` environment into the PowerShell session (see `tools/win_port/vcvars.ps1`), so no cmd wrapper
is needed.

## Run

```powershell
pwsh -File tools/win_port/serve.ps1                  # foreground TP-2 server on 127.0.0.1:8099
pwsh -File tools/win_port/serve.ps1 -Context 32768   # smaller KV capacity
pwsh -File tools/win_port/serve.ps1 -Plain           # no MTP or vision
pwsh -File tools/win_port/serve.ps1 -Status          # processes, health, VRAM
pwsh -File tools/win_port/serve.ps1 -Stop            # stop every ninfer-serve process
```

The script puts the FFmpeg and libcurl `bin` directories on `PATH` before starting the server, because
that is where the loading DLLs live. It refuses to start when a server is already running, and it only
ever manages `ninfer-serve`: no llama.cpp process is started or stopped.

Run it from a terminal you keep open. A `-Background` child belongs to the launching shell's job
object, so an agent harness that starts the server inside one tool call will kill it as soon as the call
returns (the engine then never reaches the GPU).

## Verified behavior on Windows

| Check | Linux (WSL2) | Windows |
|---|---|---|
| TP-2 plain decode | 32.8 tok/s | 32.9 tok/s |
| TP-2 MTP `--draft-tokens 2` decode | 56.8 tok/s | 56.9 tok/s |
| Image turn decode | MTP kept active | 60.4 tok/s, MTP continued (`rate=106/174`) |
| Engine load (23.7 GB artifact) | — | 35-42 s |
| Real agent traffic (OpenAI chat, streaming, tools, images) | — | served, 41-73 tok/s |

KV capacity and memory headroom on the 16 GiB cards (`--vision` enabled):

```
capacity   8192 | kv  145 MiB | free 4180 MiB (shard 0) / 3930 MiB (shard 1)
capacity 131072 | kv 2322 MiB | free 2002 MiB (shard 0) / 1580 MiB (shard 1)
```

131072 is the recommended Windows capacity: 262144, which the WSL recipe used, leaves too little VRAM
for comfort under WDDM.

### TP-2 mapped-pinned spike

Peer access is unavailable on this machine (no NVLink, `nvidia-smi topo -m` reports SYS), so TP-2 uses
the in-kernel allreduce over mapped pinned host memory. `tools/win_port/spike.ps1` exercises that
primitive outside the engine:

```
peer access: 0->1 0, 1->0 0
mapped pinned: host_a=...204A00000 view_a=...204A00000 | host_b=...20AC00000 view_b=...20AC00000
peer view of host_a from device 1: ...204A00000
     bytes   fill_ms  peer_read_ms
      4096    0.0452        0.0112   OK  (0.1 + 0.3 GiB/s)
    262144    0.0529        0.1986   OK  (4.6 + 1.2 GiB/s)
   4194304    0.6784        2.0432   OK  (5.8 + 1.9 GiB/s)
  25165824    4.0219       17.2313   OK  (5.8 + 1.4 GiB/s)
```

So `cudaHostAlloc(cudaHostAllocPortable | cudaHostAllocMapped)` plus `cudaHostGetDevicePointer` work
under WDDM, the peer device addresses the same UVA address, and cross-card reads of mapped host memory
are correct and run at the same ~1.4-1.9 GiB/s order as the WSL2 path. The end-to-end consequence is the
Linux-parity decode rate above.

## Platform differences carried by the port

| Area | Windows behavior |
|---|---|
| `src/artifact/file_io.cpp` | `CreateFileW` + `ReadFile` with an `OVERLAPPED` offset for positional reads; `FILE_FLAG_SEQUENTIAL_SCAN` for the buffered handle and `FILE_FLAG_NO_BUFFERING` for the direct one |
| `src/core/uint128.h` | 128-bit accounting without `__int128`: `_umul128` with saturating add, shift and compare |
| `src/core/host_process.{h,cpp}` | `_getpid`, `_isatty`, `localtime_s`/`gmtime_s`, and `GetConsoleScreenBufferInfo` for the progress width |
| `src/product/media_acquire/acquire.cpp` | `winsock2`/`ws2tcpip` name resolution with a one-time `WSAStartup`; links `ws2_32` |
| NVFP4 TMA kernels | the 64-byte-aligned tensor-map aggregate cannot be passed by value under the MSVC parameter ABI (C2719), so `Nvfp4TmaDescriptorStaging` uploads the descriptors into stream-ordered device memory and the kernel takes a pointer |
| `CMakeLists.txt` | MSVC needs `/Zc:preprocessor` (CUDA 13's CCCL rejects the traditional preprocessor), `/utf-8` (bundled fmt), and `NOMINMAX`/`WIN32_LEAN_AND_MEAN` |

## Limitations

- The CLI is single-GPU. This 27B NVFP4 artifact needs about 20.2 GiB of resident weights (10.1 GiB per
  TP-2 shard), so it cannot be loaded onto one 16 GiB card; use `ninfer-serve --devices 0,1` on this
  machine. The CLI binary itself builds and runs (its `--help` is exercised above), and it needs a card
  large enough for the whole model.
- The bundled libcurl is a mingw import library. It links and runs, but a native MSVC build (vcpkg) or a
  WinHTTP implementation would remove that dependency.
- Windows performance equals Linux here; the port's value is deployment convenience, not speed.
