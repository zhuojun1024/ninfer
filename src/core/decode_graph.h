#pragma once

#include <cuda_runtime.h>

#include <functional>
#include <span>

namespace ninfer {

class DecodeGraphDefinition {
public:
    DecodeGraphDefinition() = default;
    ~DecodeGraphDefinition();

    DecodeGraphDefinition(const DecodeGraphDefinition&)            = delete;
    DecodeGraphDefinition& operator=(const DecodeGraphDefinition&) = delete;
    DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept;
    DecodeGraphDefinition& operator=(DecodeGraphDefinition&& other) noexcept;

    void capture(cudaStream_t stream, const std::function<void()>& body);
    // Captures a body that launches work on several streams at once - a tensor-parallel step spans
    // two devices, so its allreduce is a paired launch on both compute streams. Every stream must
    // already be capturing while the body runs, because a launch onto a stream that is not capturing
    // cannot be recorded into another stream's graph. Each definition receives the graph of the work
    // launched on its own stream, and the streams must be in one-to-one correspondence.
    static void capture_group(std::span<DecodeGraphDefinition*> definitions,
                              std::span<cudaStream_t> streams, const std::function<void()>& body);
    [[nodiscard]] bool ready() const noexcept;
    void reset() noexcept;

private:
    friend class DecodeGraphExecutable;
    cudaGraph_t graph_ = nullptr;
};

class DecodeGraphExecutable {
public:
    DecodeGraphExecutable() = default;
    ~DecodeGraphExecutable();

    DecodeGraphExecutable(const DecodeGraphExecutable&)            = delete;
    DecodeGraphExecutable& operator=(const DecodeGraphExecutable&) = delete;
    DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept;
    DecodeGraphExecutable& operator=(DecodeGraphExecutable&& other) noexcept;

    void instantiate(const DecodeGraphDefinition& definition);
    void update(const DecodeGraphDefinition& definition);
    void upload(cudaStream_t stream);
    void launch(cudaStream_t stream);
    [[nodiscard]] bool ready() const noexcept;
    void reset() noexcept;

private:
    cudaGraphExec_t exec_ = nullptr;
};

} // namespace ninfer
