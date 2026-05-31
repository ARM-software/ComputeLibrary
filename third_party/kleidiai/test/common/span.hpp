//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <type_traits>
#include <vector>

#include "test/common/assert.hpp"

namespace kai::test {

/// Reference to a contiguous sequence of objects in the memory.
///
/// This class is similar to @ref std::span from C++20.
template <typename T>
class Span {
    template <typename U>
    friend class Span;

public:
    class Iterator {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = std::remove_cv_t<T>;
        using difference_type = ptrdiff_t;
        using pointer = T*;
        using reference = T&;

        constexpr Iterator(T* ptr) : m_ptr(ptr) {
        }

        constexpr T& operator*() const noexcept {
            return *m_ptr;
        }

        constexpr T* operator->() const noexcept {
            return m_ptr;
        }

        constexpr Iterator& operator++() noexcept {
            ++m_ptr;
            return *this;
        }

        constexpr Iterator operator++(int) noexcept {
            Iterator it = *this;
            ++m_ptr;
            return it;
        }

        constexpr Iterator& operator--() noexcept {
            --m_ptr;
            return *this;
        }

        constexpr Iterator operator--(int) noexcept {
            Iterator it = *this;
            --m_ptr;
            return it;
        }

        constexpr Iterator& operator+=(size_t len) noexcept {
            m_ptr += len;
            return *this;
        }

        constexpr Iterator& operator-=(size_t len) noexcept {
            m_ptr -= len;
            return *this;
        }

        [[nodiscard]] constexpr Iterator operator+(size_t len) const noexcept {
            return {m_ptr + len};
        }

        [[nodiscard]] friend constexpr Iterator operator+(size_t len, Iterator it) noexcept {
            return {it.m_ptr + len};
        }

        [[nodiscard]] constexpr Iterator operator-(size_t len) const noexcept {
            return {m_ptr - len};
        }

        [[nodiscard]] constexpr ptrdiff_t operator-(Iterator other) const noexcept {
            return m_ptr - other.m_ptr;
        }

        [[nodiscard]] constexpr T& operator[](size_t idx) const noexcept {
            return m_ptr[idx];
        }

        [[nodiscard]] constexpr bool operator==(Iterator other) const noexcept {
            return m_ptr == other.m_ptr;
        }

        [[nodiscard]] constexpr bool operator!=(Iterator other) const noexcept {
            return m_ptr != other.m_ptr;
        }

        [[nodiscard]] constexpr bool operator>(Iterator other) const noexcept {
            return m_ptr > other.m_ptr;
        }

        [[nodiscard]] constexpr bool operator<(Iterator other) const noexcept {
            return m_ptr < other.m_ptr;
        }

        [[nodiscard]] constexpr bool operator>=(Iterator other) const noexcept {
            return m_ptr >= other.m_ptr;
        }

        [[nodiscard]] constexpr bool operator<=(Iterator other) const noexcept {
            return m_ptr <= other.m_ptr;
        }

    private:
        T* m_ptr;
    };

    /// Creates an empty span.
    constexpr Span() noexcept : m_ptr(nullptr), m_size(0) {
    }

    /// Creates a span refering to the specified sequence.
    ///
    /// @param[in] first The iterator to the first element of the sequence.
    /// @param[in] count The number of elements.
    template <typename It>
    constexpr Span(It first, size_t count) noexcept : m_ptr(first), m_size(count) {
    }

    /// Creates a span refering to the specified array.
    ///
    /// @param[in] arr The array to be refered.
    template <typename U, size_t N>
    constexpr Span(const std::array<U, N>& arr) noexcept : m_ptr(arr.data()), m_size(arr.size()) {
    }

    /// Creates a span refering to the specified array.
    ///
    /// @param[in] arr The array to be refered.
    template <typename U, size_t N>
    constexpr Span(std::array<U, N>& arr) noexcept : m_ptr(arr.data()), m_size(arr.size()) {
    }

    /// Creates a span refering to the specified vector.
    ///
    /// @param[in] vec The vector to be refered.
    template <typename U>
    constexpr Span(const std::vector<U>& vec) noexcept : m_ptr(vec.data()), m_size(vec.size()) {
    }

    /// Creates a span refering to the specified vector.
    ///
    /// @param[in] vec The vector to be refered.
    template <typename U>
    constexpr Span(std::vector<U>& vec) noexcept : m_ptr(vec.data()), m_size(vec.size()) {
    }

    /// Creates a span refering to the specified initializer list.
    ///
    /// @param[in] il The initializer list to be refered.
    constexpr Span(std::initializer_list<T> il) noexcept : m_ptr(std::data(il)), m_size(il.size()) {
    }

    /// Creates a spen refering to the specified span.
    ///
    /// @param[in] other The span to be refered.
    template <typename U>
    constexpr Span(const Span<U>& other) noexcept : m_ptr(other.m_ptr), m_size(other.m_size) {
    }

    /// Destructor.
    ~Span() = default;

    /// Copy constructor.
    constexpr Span(const Span&) noexcept = default;

    /// Copy assignment.
    constexpr Span& operator=(const Span&) noexcept = default;

    /// Move constructor.
    constexpr Span(Span&&) noexcept = default;

    /// Move assignment.
    constexpr Span& operator=(Span&&) noexcept = default;

    /// Gets a forward iterator to the beginning of the span.
    [[nodiscard]] constexpr Iterator begin() const noexcept {
        return {m_ptr};
    }

    /// Gets a forward iterator to the end of the span.
    [[nodiscard]] constexpr Iterator end() const noexcept {
        return {m_ptr + m_size};
    }

    /// Gets the first element.
    [[nodiscard]] constexpr T& front() const {
        return m_ptr[0];
    }

    /// Gets the last element.
    [[nodiscard]] constexpr T& back() const {
        return m_ptr[m_size - 1];
    }

    /// Gets the element at the specified index with bounds checking.
    [[nodiscard]] constexpr T& at(size_t idx) const {
        KAI_TEST_ASSERT(idx < m_size);
        return m_ptr[idx];
    }

    /// Gets the element at the specified index.
    [[nodiscard]] constexpr T& operator[](size_t idx) const {
        return m_ptr[idx];
    }

    /// Gets the pointer to the data.
    [[nodiscard]] constexpr T* data() const noexcept {
        return m_ptr;
    }

    /// Gets the number of elements.
    [[nodiscard]] constexpr size_t size() const noexcept {
        return m_size;
    }

    /// Gets a value indicating whether the span is empty.
    [[nodiscard]] constexpr bool empty() const noexcept {
        return m_size == 0;
    }

    /// Gets a sub-view of the span.
    [[nodiscard]] constexpr Span subspan(size_t offset) const {
        KAI_TEST_ASSERT(offset <= m_size);
        return {m_ptr + offset, m_size - offset};
    }

    /// Gets a sub-view of the span.
    [[nodiscard]] constexpr Span subspan(size_t offset, size_t count) const {
        KAI_TEST_ASSERT(offset + count <= m_size);
        return {m_ptr + offset, count};
    }

private:
    T* m_ptr;
    size_t m_size;
};

}  // namespace kai::test
