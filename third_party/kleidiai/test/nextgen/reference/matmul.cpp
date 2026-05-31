//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/reference/matmul.hpp"

#include <cstddef>

#include "test/common/assert.hpp"
#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"
#include "test/common/memory.hpp"
#include "test/common/round.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/functions/fused_mul_add.hpp"

namespace kai::test {

namespace {

template <typename T>
[[nodiscard]] Buffer matmul_nt_t(
    size_t shape_m, size_t shape_n, size_t shape_k, Span<const std::byte> lhs, Span<const std::byte> rhs) {
    Buffer dst(shape_m * round_up_division(shape_n * size_in_bits<T>, 8), 0);

    for (size_t row = 0; row < shape_m; ++row) {
        for (size_t col = 0; col < shape_n; ++col) {
            T acc = static_cast<T>(0);

            for (size_t depth = 0; depth < shape_k; ++depth) {
                const T lhs_value = read_2d<T>(lhs, shape_k, row, depth);
                const T rhs_value = read_2d<T>(rhs, shape_k, col, depth);

                acc = fused_mul_add<T>(lhs_value, rhs_value, acc);
            }

            write_2d<T>(dst.view(), shape_n, row, col, acc);
        }
    }

    return dst;
}

}  // namespace

MatMulFn make_matmul_nt_t(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:
            return matmul_nt_t<float>;

        default:
            KAI_TEST_ERROR("Not implemented.");
    }
}

}  // namespace kai::test
