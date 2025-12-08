//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <ostream>
#include <string_view>
#include <utility>
#include <vector>

#include "test/common/assert.hpp"
#include "test/common/buffer.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/format.hpp"

namespace kai::test {

/// Multidimensional data slot.
class Tensor {
public:
    /// Gets the size of the multidimensional array.
    [[nodiscard]] Span<const size_t> shape() const {
        return m_shape;
    }

    /// Sets the shape.
    ///
    /// A slot cannot change its shape. If the shape is already non-zero,
    /// the new shape must be the same as the existing shape
    /// to make sure all components in the test environment have the same
    /// expectation for this slot.
    Tensor& set_shape(Span<const size_t> shape) {
        if (m_shape.empty()) {
            KAI_TEST_ASSERT_MSG(m_data.size() == 0, "The slot must be empty when its shape is setup.");
            m_shape = std::vector(shape.begin(), shape.end());
        } else {
            KAI_TEST_ASSERT_MSG(
                std::equal(m_shape.begin(), m_shape.end(), shape.begin(), shape.end()),
                "The new shape must be the same as the existing shape.");
        }

        return *this;
    }

    /// Gets the data format.
    [[nodiscard]] const Poly<Format>& format() const {
        return m_format.value();
    }

    /// Sets the data format.
    ///
    /// A slot cannot change its data format. If the data format is already known,
    /// the new format must be the same as the existing one
    /// to make sure all components in the test environment have the same
    /// expectation for this slot.
    Tensor& set_format(const Poly<Format>& format) {
        if (!m_format.has_value()) {
            KAI_TEST_ASSERT_MSG(m_data.size() == 0, "The slot must be empty when its format is setup.");
            m_format = format;
        } else {
            KAI_TEST_ASSERT_MSG(
                *m_format.value() == *format, "The new format must be the same as the existing format.");
        }

        return *this;
    }

    /// Gets the data.
    [[nodiscard]] Span<const std::byte> data() const {
        return m_data;
    }

    /// Gets the data.
    [[nodiscard]] Span<std::byte> data() {
        return m_data;
    }

    /// Gets the data.
    [[nodiscard]] const std::byte* data_ptr() const {
        return m_data.data();
    }

    /// Gets the data.
    [[nodiscard]] std::byte* data_ptr() {
        return m_data.data();
    }

    /// Gets the value in custom format.
    template <typename T>
    [[nodiscard]] const T& value() const {
        KAI_TEST_ASSERT_MSG(!m_format.has_value(), "This method only works on custom data.");
        KAI_TEST_ASSERT_MSG(m_data.size() == sizeof(T), "The data size is incorrect.");

        return *reinterpret_cast<const T*>(data_ptr());
    }

    /// Gets the value in custom format.
    template <typename T>
    [[nodiscard]] T& value() {
        KAI_TEST_ASSERT_MSG(!m_format.has_value(), "This method only works on custom data.");
        KAI_TEST_ASSERT_MSG(m_data.size() == sizeof(T), "The data size is incorrect.");

        return *reinterpret_cast<T*>(data_ptr());
    }

    /// Sets the value in custom format.
    template <typename T>
    void set_value(T&& value) {
        KAI_TEST_ASSERT_MSG(!m_format.has_value(), "This method only works on custom data.");
        if (m_shape.empty()) {
            KAI_TEST_ASSERT_MSG(
                m_data.size() == 0, "If the shape is not specified, the data cannot be already allocated.");

            set_shape({sizeof(T)});
            allocate();
        } else {
            KAI_TEST_ASSERT_MSG(m_shape.size() == 1 && m_shape.at(0) == sizeof(T), "The shape is incorrect.");
            KAI_TEST_ASSERT_MSG(m_data.size() == sizeof(T), "The data size is incorrect.");
        }

        this->value<T>() = std::forward<T>(value);
    }

    /// Allocates and resets the data buffer.
    void allocate() {
        const size_t size = compute_size();
        m_data = Buffer(size, 0);
    }

    /// Sets the data buffer.
    ///
    /// The new data buffer must have the right size.
    void set_data(Buffer&& buffer) {
        [[maybe_unused]] const size_t size = compute_size();
        KAI_TEST_ASSERT_MSG(buffer.size() == size, "New data buffer must have the right size.");
        m_data = std::move(buffer);
    }

private:
    [[nodiscard]] size_t compute_size() const {
        if (m_format.has_value()) {
            return m_format.value()->compute_size(m_shape);
        }

        // If the data format is not available, this slot is used to store custom data
        // rather than multidimensional array. The first element of the shape
        // is the size of the data being stored.
        KAI_TEST_ASSERT_MSG(
            m_shape.size() == 1, "Custom data slot must use the shape to store the size of entire data.");
        return m_shape.at(0);
    }

    std::vector<size_t> m_shape;
    std::optional<Poly<Format>> m_format;
    Buffer m_data;
};

}  // namespace kai::test
