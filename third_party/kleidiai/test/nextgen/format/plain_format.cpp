//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/format/plain_format.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <ostream>
#include <random>

#include "test/common/assert.hpp"
#include "test/common/buffer.hpp"
#include "test/common/compare.hpp"
#include "test/common/data_type.hpp"
#include "test/common/round.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/common/random.hpp"
#include "test/nextgen/format/format.hpp"
#include "test/nextgen/reference/compare.hpp"
#include "test/nextgen/reference/print.hpp"
#include "test/reference/fill.hpp"

namespace kai::test {

size_t PlainFormat::compute_offset(Span<const size_t> shape, Span<const size_t> indices) const {
    KAI_TEST_ASSERT(shape.size() > 0);
    KAI_TEST_ASSERT(shape.size() == indices.size());

    const size_t num_dims = shape.size();
    size_t stride = round_up_division(shape.at(num_dims - 1) * data_type_size_in_bits(m_dtype), 8);
    size_t offset = indices.at(num_dims - 1) * data_type_size_in_bits(m_dtype) / 8;
    KAI_TEST_ASSERT(indices.at(num_dims - 1) * data_type_size_in_bits(m_dtype) % 8 == 0);

    if (num_dims > 1) {
        size_t dim = num_dims - 2;

        while (true) {
            offset += indices.at(dim) * stride;
            stride *= shape.at(dim);

            if (dim == 0) {
                break;
            }

            --dim;
        }
    }

    return offset;
}

size_t PlainFormat::compute_size(Span<const size_t> shape) const {
    if (shape.empty()) {
        return 0;
    }

    const size_t row_size = round_up_division(shape.at(shape.size() - 1) * data_type_size_in_bits(m_dtype), 8);
    const size_t num_rows = std::accumulate(shape.begin(), shape.end() - 1, 1, std::multiplies<>());
    const size_t size = row_size * num_rows;

    return size;
}

Buffer PlainFormat::generate_random(Span<const size_t> shape, Rng& rng) const {
    const size_t len = compute_size(shape) * 8 / data_type_size_in_bits(m_dtype);
    const uint32_t seed =
        std::uniform_int_distribution<uint32_t>()(rng);  // REVISIT: Use the random number generator directly.

    switch (m_dtype) {
        case DataType::FP32:

            return fill_random<float>(len, seed);

        default:
            KAI_TEST_ERROR("Not supported!");
    }
}

Buffer PlainFormat::pack(Span<const size_t> shape, Span<const Span<const std::byte>> buffers) const {
    KAI_TEST_ASSERT_MSG(buffers.size() == 1, "Plain format only has 1 data component.");

    const Span<const std::byte> data = buffers.at(0);
    const size_t size = compute_size(shape);
    KAI_TEST_ASSERT_MSG(data.size() == size, "The data buffer must have the right size.");

    Buffer packed_buffer(size);
    std::copy(data.begin(), data.end(), packed_buffer.data());

    return packed_buffer;
}

bool PlainFormat::compare(
    Span<const size_t> shape, Span<const size_t> tile_coords, Span<const size_t> tile_shape,
    Span<const std::byte> imp_buffer, Span<const std::byte> ref_buffer, MismatchHandler& handler) const {
    KAI_TEST_ASSERT_MSG(shape.size() == 2, "Only 2D array is supported.");

    const CompareFn compare_fn = make_compare_plain_2d(m_dtype);
    const size_t num_checks = compare_fn(
        shape, tile_coords, tile_shape, imp_buffer, ref_buffer,
        [](std::ostream& os, Span<const size_t> indices) {
            os << "Mismatch at row " << indices.at(0) << ", col " << indices.at(1);
        },
        handler);

    return handler.success(num_checks);
}

void PlainFormat::print(std::ostream& os, Span<const size_t> shape, Span<const std::byte> data) const {
    if (shape.empty()) {
        os << "None";
    } else {
        const PrintFn print_fn = make_print_array(m_dtype);
        print_fn(os, shape, data, 0);
    }
}

bool PlainFormat::operator==(const Format& other) const {
    const auto* rhs = dynamic_cast<const PlainFormat*>(&other);
    return rhs != nullptr && m_dtype == rhs->m_dtype;
}

}  // namespace kai::test
