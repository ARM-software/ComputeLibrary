//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <type_traits>
#include <utility>

#include "kai/kai_common.h"
#include "test/common/buffer.hpp"
#include "test/common/memory.hpp"
namespace kai::test {

class DataFormat;

/// Creates a new matrix filled with random data.
///
/// @param[in] height Number of rows.
/// @param[in] width Number of columns.
/// @param[in] format Data format.
/// @param[in] seed Random seed.
///
/// @return The data buffer for the matrix.
Buffer fill_matrix_random(size_t height, size_t width, const DataFormat& format, uint32_t seed);

/// Creates a new data buffer filled with random data.
///
/// @tparam Value The data type.
///
/// @param[in] length The number of elements.
/// @param[in] seed The random seed.
///
/// @return The data buffer.
template <typename Value>
Buffer fill_random(size_t length, uint32_t seed);

/// Creates a new matrix filled with data produced by a generator function.
///
/// @tparam T Element type.
/// @tparam Generator Callable returning values convertible to `T`.
///
/// @param[in] height Number of rows.
/// @param[in] width Number of columns.
/// @param[in] gen Generator function or functor.
///
/// @return The data buffer for the matrix.
template <typename T, typename Generator>
Buffer fill_matrix_raw(size_t height, size_t width, Generator&& gen) {
    KAI_ASSUME_ALWAYS(width * size_in_bits<T> % 8 == 0);
    const auto row_bytes = width * size_in_bits<T> / 8;

    Buffer data(height * row_bytes);
    auto* ptr = reinterpret_cast<T*>(data.data());
    auto&& generator = std::forward<Generator>(gen);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            write_array<T>(ptr, y * width + x, generator(y, x));
        }
    }

    return data;
}

/// Convenience overload to maintain the legacy std::function signature.
template <typename T>
Buffer fill_matrix_raw(size_t height, size_t width, std::function<T(size_t, size_t)> gen) {
    // Wrap std::function into a generic generator to prevent self-recursion.
    return fill_matrix_raw<T>(height, width, [&gen](size_t y, size_t x) { return gen(y, x); });
}

/// Creates a new matrix using a generator that returns a Buffer.
///
/// @tparam T Element type.
/// @tparam Generator Callable returning `Buffer`.
///
/// @param[in] height Number of rows.
/// @param[in] width Number of columns.
/// @param[in] generator Generator instance.
///
/// @return The data buffer for the matrix.
template <typename Generator>
Buffer fill_matrix_generate(size_t height, size_t width, const Generator& generator) {
    auto buffer = generator(height, width);
    static_assert(std::is_same_v<std::decay_t<decltype(buffer)>, Buffer>, "Generator must return Buffer");

    return buffer;
}

}  // namespace kai::test
