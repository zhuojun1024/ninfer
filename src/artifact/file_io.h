#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

namespace ninfer::artifact {

// Native file handle: a POSIX descriptor or a Windows HANDLE. A Windows handle does not fit an int,
// so only the private storage follows the platform; the public contract is unchanged.
#if defined(_WIN32)
using NativeFileHandle                  = void*;
inline constexpr NativeFileHandle kInvalidFileHandle = nullptr;
#else
using NativeFileHandle                  = int;
inline constexpr NativeFileHandle kInvalidFileHandle = -1;
#endif

// Direct reads require aligned offsets and buffers. A short final direct block is allowed;
// read_exact always requires the complete requested byte range.
class InputFile {
public:
    explicit InputFile(std::filesystem::path path);
    ~InputFile();
    InputFile(const InputFile&)            = delete;
    InputFile& operator=(const InputFile&) = delete;

    [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }

    void read_exact(std::uint64_t offset, std::span<std::byte> destination) const;
    [[nodiscard]] std::size_t read_direct(std::uint64_t offset,
                                          std::span<std::byte> destination) const;

private:
    std::filesystem::path path_;
    NativeFileHandle fd_                = kInvalidFileHandle;
    mutable NativeFileHandle direct_fd_ = kInvalidFileHandle;
    std::uint64_t bytes_                = 0;
};

} // namespace ninfer::artifact
