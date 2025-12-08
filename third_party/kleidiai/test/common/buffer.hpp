//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include "test/common/span.hpp"

namespace kai::test {

/// Buffer is a high-level abstraction for a block of memory.
///
/// The class performs dynamic memory allocation and management in an opaque manner. The underlying memory resource can
/// be requested using the familiar @ref Buffer::data() method and interacted with using @ref
/// kai::test::read_array<T>() and @ref kai::test::write_array<T>() utilities.
///
/// Buffer comes with protection mechanisms defined by @ref BufferProtectionPolicy. These are enabled by setting the
/// KAI_TEST_BUFFER_POLICY environment variable, for example:
///     KAI_TEST_BUFFER_POLICY=PROTECT_UNDERFLOW to enable @ref BufferProtectionPolicy::ProtectUnderflow.
///     KAI_TEST_BUFFER_POLICY=PROTECT_OVERFLOW to enable @ref BufferProtectionPolicy::ProtectOverflow.
///
class Buffer {
    // Handle to the underlying memory resource and its deleter
    using handle = std::unique_ptr<void, std::function<void(void*)>>;

public:
    Buffer() = default;
    explicit Buffer(size_t size);
    Buffer(size_t size, uint8_t init_value);

    Buffer(const Buffer& other) = delete;
    Buffer(Buffer&& other) noexcept = default;
    Buffer& operator=(const Buffer& other) = delete;
    Buffer& operator=(Buffer&& other) noexcept = default;

    ~Buffer() = default;

    /// Gets the base memory address of the user buffer.
    ///
    /// @return Base memory address of the user buffer.
    [[nodiscard]] std::byte* data() const {
        return static_cast<std::byte*>(m_buffer.get()) + m_user_buffer_offset;
    }

    /// Gets a view of the data.
    operator Span<const std::byte>() const {
        return {data(), size()};
    }

    /// Gets a view of the data.
    operator Span<std::byte>() {
        return {data(), size()};
    }

    /// Gets a view of the data.
    [[nodiscard]] Span<const std::byte> view() const {
        return {data(), size()};
    }

    /// Gets a view of the data.
    [[nodiscard]] Span<std::byte> view() {
        return {data(), size()};
    }

    /// Gets the size of the user buffer.
    ///
    /// Depending on the @ref BufferProtectionPolicy policy enabled, the actual size of memory allocated may be larger.
    /// However, this function guarantees to always provide the size of the user buffer only.
    ///
    /// @return Size of the user buffer in bytes.
    [[nodiscard]] size_t size() const {
        return m_user_buffer_size;
    }

    static constexpr const char* buffer_policy_env_name = "KAI_TEST_BUFFER_POLICY";

private:
    /// Buffer can be protected with one of the following protection policies:
    ///   - @ref BufferProtectionPolicy::None              No protection mechanisms are enabled.
    ///   - @ref BufferProtectionPolicy::ProtectUnderflow  Memory equal to the size of the user buffer rounded to the
    ///                                                    nearest whole page plus adjacent guard pages is allocated,
    ///                                                    and the user buffer is aligned to the end of the head guard
    ///                                                    page thus detecting whenever a buffer underflow occurs.
    ///   - @ref BufferProtectionPolicy::ProtectOverflow   Same as above, but now the edge of the user buffer is aligned
    ///                                                    to the start of the tail guard page thus detecting whenever a
    ///                                                    buffer overflow occurs.
    enum class BufferProtectionPolicy : uint8_t {
        None = 0,
        ProtectUnderflow = 1,
        ProtectOverflow = 2,
    };

    /// Naively allocate memory.
    void allocate();

#if defined(__linux__) || defined(__APPLE__)
    /// Allocate memory with adjacent guard pages.
    void allocate_with_guard_pages();
#endif  // defined(__linux__) || defined(__APPLE__)

    handle m_buffer = nullptr;

    size_t m_user_buffer_size = 0;
    size_t m_user_buffer_offset = 0;

    BufferProtectionPolicy m_protection_policy = BufferProtectionPolicy::None;
};

}  // namespace kai::test
