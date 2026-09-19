#include "artifact/file_io.h"

#include "artifact/framing.h"
#include "artifact/schema.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <utility>

#if defined(_WIN32)
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#else
#    include <fcntl.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

namespace ninfer::artifact {
namespace {

#if defined(_WIN32)

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw ArtifactError(path.string() + ": " + operation + ": Windows error " +
                        std::to_string(static_cast<unsigned long>(::GetLastError())));
}

// An OVERLAPPED offset on a synchronous handle gives the same positional-read contract as pread:
// read_exact never depends on shared file-position state, so concurrent readers stay independent.
std::int64_t positional_read(NativeFileHandle handle, std::uint64_t offset, void* destination,
                             std::size_t count) {
    if (count > std::numeric_limits<DWORD>::max()) { return -1; }
    OVERLAPPED overlapped{};
    overlapped.Offset     = static_cast<DWORD>(offset & 0xFFFFFFFFULL);
    overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);
    DWORD read            = 0;
    if (::ReadFile(handle, destination, static_cast<DWORD>(count), &read, &overlapped) == 0) {
        return -1;
    }
    return static_cast<std::int64_t>(read);
}

NativeFileHandle open_handle(const std::filesystem::path& path, bool direct) {
    const DWORD flags = FILE_ATTRIBUTE_NORMAL | (direct ? FILE_FLAG_NO_BUFFERING
                                                        : FILE_FLAG_SEQUENTIAL_SCAN);
    // Reading must not lock the artifact. POSIX readers stay valid when another process rewrites,
    // replaces or deletes the file, and the reader and writer-interop checks depend on that.
    const DWORD share = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
    const HANDLE handle = ::CreateFileW(path.c_str(), GENERIC_READ, share, nullptr, OPEN_EXISTING,
                                        flags, nullptr);
    return handle == INVALID_HANDLE_VALUE ? kInvalidFileHandle : handle;
}

#else

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw ArtifactError(path.string() + ": " + operation + ": " + std::strerror(errno));
}

off_t file_offset(std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw ArtifactError("file offset exceeds positional I/O range");
    }
    return static_cast<off_t>(offset);
}

std::int64_t positional_read(NativeFileHandle handle, std::uint64_t offset, void* destination,
                             std::size_t count) {
    for (;;) {
        const auto read = ::pread(handle, destination, count, file_offset(offset));
        if (read < 0 && errno == EINTR) { continue; }
        return static_cast<std::int64_t>(read);
    }
}

NativeFileHandle open_handle(const std::filesystem::path& path, bool direct) {
    return ::open(path.c_str(), O_RDONLY | O_CLOEXEC | (direct ? O_DIRECT : 0));
}

#endif

} // namespace

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
    fd_ = open_handle(path_, false);
    if (fd_ == kInvalidFileHandle) { fail(path_, "open"); }

#if defined(_WIN32)
    LARGE_INTEGER size{};
    if (::GetFileSizeEx(fd_, &size) == 0 || size.QuadPart < 0 ||
        ::GetFileType(fd_) != FILE_TYPE_DISK) {
        const DWORD error = ::GetLastError();
        ::CloseHandle(fd_);
        fd_ = kInvalidFileHandle;
        ::SetLastError(error);
        fail(path_, "stat");
    }
    bytes_ = static_cast<std::uint64_t>(size.QuadPart);
#else
    struct stat status {};
    if (::fstat(fd_, &status) != 0) {
        const auto error = errno;
        ::close(fd_);
        fd_   = -1;
        errno = error;
        fail(path_, "fstat");
    }
    if (status.st_size < 0 || !S_ISREG(status.st_mode)) {
        ::close(fd_);
        fd_ = -1;
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = static_cast<std::uint64_t>(status.st_size);
#endif
}

InputFile::~InputFile() {
    for (NativeFileHandle handle : {direct_fd_, fd_}) {
        if (handle == kInvalidFileHandle) { continue; }
#if defined(_WIN32)
        ::CloseHandle(handle);
#else
        ::close(handle);
#endif
    }
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        const auto read  = positional_read(fd_, offset, destination.data(), count);
        if (read < 0) { fail(path_, "read"); }
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += static_cast<std::uint64_t>(read);
        destination = destination.subspan(static_cast<std::size_t>(read));
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw ArtifactError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
    if (direct_fd_ == kInvalidFileHandle) {
        direct_fd_ = open_handle(path_, true);
        if (direct_fd_ == kInvalidFileHandle) { fail(path_, "open direct"); }
    }
    const auto read = positional_read(direct_fd_, offset, destination.data(), destination.size());
    if (read < 0) { fail(path_, "direct read"); }
    return static_cast<std::size_t>(read);
}

} // namespace ninfer::artifact
