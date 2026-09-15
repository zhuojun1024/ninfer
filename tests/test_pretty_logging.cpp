#include "product/logging/logging.h"
#include "product/logging/startup_log.h"

#include <spdlog/logger.h>

#if defined(_WIN32)
#    include <fcntl.h>
#    include <io.h>
#else
#    include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>

namespace {

#if defined(_WIN32)

// The MSVC CRT keeps stderr on descriptor 2, but pipes, duplication and positional reads carry the
// underscore names. The pipe is binary so the captured bytes match the POSIX harness exactly.
constexpr int kStderrFd = 2;
int pipe_open(int descriptors[2]) { return ::_pipe(descriptors, 65536, _O_BINARY); }
int duplicate_fd(int descriptor) { return ::_dup(descriptor); }
int duplicate_onto(int from, int to) { return ::_dup2(from, to); }
void close_fd(int descriptor) { (void)::_close(descriptor); }
std::ptrdiff_t read_fd(int descriptor, void* buffer, std::size_t bytes) {
    return static_cast<std::ptrdiff_t>(::_read(descriptor, buffer, static_cast<unsigned>(bytes)));
}

#else

constexpr int kStderrFd = STDERR_FILENO;
int pipe_open(int descriptors[2]) { return ::pipe(descriptors); }
int duplicate_fd(int descriptor) { return ::dup(descriptor); }
int duplicate_onto(int from, int to) { return ::dup2(from, to); }
void close_fd(int descriptor) { ::close(descriptor); }
std::ptrdiff_t read_fd(int descriptor, void* buffer, std::size_t bytes) {
    return static_cast<std::ptrdiff_t>(::read(descriptor, buffer, bytes));
}

#endif

class StderrCapture {
public:
    StderrCapture() {
        if (pipe_open(pipe_) != 0) { throw std::runtime_error(std::strerror(errno)); }
        saved_ = duplicate_fd(kStderrFd);
        if (saved_ < 0 || duplicate_onto(pipe_[1], kStderrFd) < 0) {
            throw std::runtime_error(std::strerror(errno));
        }
#if defined(_WIN32)
        (void)::_setmode(kStderrFd, _O_BINARY);
#endif
        close_fd(pipe_[1]);
        pipe_[1] = -1;
    }

    ~StderrCapture() {
        if (saved_ >= 0) {
            (void)duplicate_onto(saved_, kStderrFd);
            close_fd(saved_);
        }
        if (pipe_[0] >= 0) { close_fd(pipe_[0]); }
    }

    std::string finish() {
        std::fflush(stderr);
        if (duplicate_onto(saved_, kStderrFd) < 0) { throw std::runtime_error(std::strerror(errno)); }
        close_fd(saved_);
        saved_ = -1;

        std::string output;
        std::array<char, 4096> buffer{};
        for (;;) {
            const std::ptrdiff_t count = read_fd(pipe_[0], buffer.data(), buffer.size());
            if (count == 0) { break; }
            if (count < 0) {
                if (errno == EINTR) { continue; }
                throw std::runtime_error(std::strerror(errno));
            }
            output.append(buffer.data(), static_cast<std::size_t>(count));
        }
        close_fd(pipe_[0]);
        pipe_[0] = -1;
        return output;
    }

private:
    int pipe_[2]{-1, -1};
    int saved_ = -1;
};

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

std::size_t line_count(std::string_view value) {
    return static_cast<std::size_t>(std::count(value.begin(), value.end(), '\n'));
}

} // namespace

int main() {
    int failures = 0;
    std::string service_output;
    {
        StderrCapture capture;
        {
            ninfer::product::LoggingRuntime logging(
                {.logger_name  = "ninfer-serve",
                 .color        = ninfer::product::LogColorMode::Auto,
                 .presentation = ninfer::product::LogPresentation::Service});
            logging.logger()->info("throughput | sample");
            logging.flush();
        }
        service_output = capture.finish();
    }
    failures += check(
        std::regex_match(
            service_output,
            std::regex(
                R"(^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}  INFO  throughput \| sample\n$)")),
        "service pretty prefix mismatch");
    failures += check(service_output.find("ninfer-serve") == std::string::npos,
                      "service pretty output repeated the executable name");
    failures += check(service_output.find("\x1b[") == std::string::npos,
                      "redirected service output contains ANSI escapes");

    std::string startup_output;
    {
        StderrCapture capture;
        {
            ninfer::product::LoggingRuntime logging(
                {.logger_name  = "ninfer-serve",
                 .color        = ninfer::product::LogColorMode::Never,
                 .presentation = ninfer::product::LogPresentation::Service});
            ninfer::product::StartupLogRenderer startup(logging);
            ninfer::StartupObserver observer = startup.observer();
            observer.callback({.phase  = ninfer::StartupPhase::EngineStartup,
                               .status = ninfer::StartupStatus::Begin});
            observer.callback({.phase  = ninfer::StartupPhase::CudaInitialize,
                               .status = ninfer::StartupStatus::Begin});
            observer.callback({.phase      = ninfer::StartupPhase::CudaInitialize,
                               .status     = ninfer::StartupStatus::Complete,
                               .elapsed_ns = 1'000'000'000});
            observer.callback({.phase         = ninfer::StartupPhase::WeightsMaterialize,
                               .status        = ninfer::StartupStatus::Begin,
                               .progress_unit = ninfer::StartupProgressUnit::Bytes,
                               .total         = 16ULL << 30});
            observer.callback({.phase         = ninfer::StartupPhase::WeightsMaterialize,
                               .status        = ninfer::StartupStatus::Complete,
                               .progress_unit = ninfer::StartupProgressUnit::Bytes,
                               .current       = 16ULL << 30,
                               .total         = 16ULL << 30,
                               .elapsed_ns    = 2'000'000'000});
            observer.callback({.phase      = ninfer::StartupPhase::EngineStartup,
                               .status     = ninfer::StartupStatus::Complete,
                               .elapsed_ns = 3'000'000'000});
            startup.engine_ready({.model_name           = "qwen3.6-27b",
                                  .weight_formats       = {"q4_g64_fp16", "q8_g32_fp16"},
                                  .host_to_device_bytes = 16ULL << 30});
            logging.flush();
        }
        startup_output = capture.finish();
    }
    failures += check(
        line_count(startup_output) == 4 &&
            startup_output.find("starting engine") != std::string::npos &&
            startup_output.find("loading weights | 16.0 GiB") != std::string::npos &&
            startup_output.find("weights ready | 16.0 GiB | 2.0s | 8.00 GiB/s") !=
                std::string::npos &&
            startup_output.find("engine ready | qwen3.6-27b | total 3.0s | weights 16.0 GiB") !=
                std::string::npos &&
            startup_output.find("CUDA initialized") == std::string::npos,
        "normal startup pretty output is noisy or incomplete");

    std::string tool_output;
    {
        StderrCapture capture;
        {
            ninfer::product::LoggingRuntime logging(
                {.logger_name  = "ninfer",
                 .color        = ninfer::product::LogColorMode::Never,
                 .presentation = ninfer::product::LogPresentation::Tool});
            logging.logger()->info("engine ready");
            logging.logger()->error("failed");
            logging.flush();
        }
        tool_output = capture.finish();
    }
    failures +=
        check(tool_output == "engine ready\nerror: failed\n", "tool pretty prefix mismatch");
    return failures == 0 ? 0 : 1;
}
