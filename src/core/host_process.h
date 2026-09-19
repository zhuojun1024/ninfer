#pragma once

// Host-process queries that differ between POSIX and Windows and are shared by the artifact, runtime
// and product layers. Only the CRT headers are needed: pulling windows.h into a widely included
// header would leak its macros into every consumer.

#include <cstdio>
#include <ctime>

#if defined(_WIN32)
#    include <io.h>
#    include <process.h>
#else
#    include <unistd.h>
#endif

namespace ninfer {

[[nodiscard]] inline long host_process_id() noexcept {
#if defined(_WIN32)
    return static_cast<long>(::_getpid());
#else
    return static_cast<long>(::getpid());
#endif
}

[[nodiscard]] inline bool host_stderr_is_interactive() noexcept {
#if defined(_WIN32)
    return ::_isatty(::_fileno(stderr)) != 0;
#else
    return ::isatty(STDERR_FILENO) != 0;
#endif
}

inline void host_localtime(std::time_t seconds, std::tm& local) noexcept {
#if defined(_WIN32)
    (void)::localtime_s(&local, &seconds);
#else
    (void)::localtime_r(&seconds, &local);
#endif
}

inline void host_gmtime(std::time_t seconds, std::tm& utc) noexcept {
#if defined(_WIN32)
    (void)::gmtime_s(&utc, &seconds);
#else
    (void)::gmtime_r(&seconds, &utc);
#endif
}

} // namespace ninfer
