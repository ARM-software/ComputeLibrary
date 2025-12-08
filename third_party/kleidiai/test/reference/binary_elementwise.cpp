//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/reference/binary_elementwise.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "kai/kai_common.h"
#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"
#include "test/common/float16.hpp"
#include "test/common/int4.hpp"
#include "test/common/memory.hpp"

namespace kai::test {

namespace {

/// Binary element-wise operator.
enum class BinaryElementwiseOperator : uint32_t {
    ADD,  ///< Addition.
    SUB,  ///< Subtraction.
    MUL,  ///< Multiplication.
    DIV,  ///< Division.
};

/// Scalar binary element-wise function.
///
/// @tparam op Binary element-wise operator to perform.
/// @tparam T Data type.
///
/// @param[in] lhs LHS operand.
/// @param[in] rhs RHS operand.
///
/// @return The result of the operation.
template <const BinaryElementwiseOperator op, typename T>
T scalar_binary_elementwise(T lhs, T rhs) {
    if constexpr (op == BinaryElementwiseOperator::ADD) {
        return lhs + rhs;
    } else if constexpr (op == BinaryElementwiseOperator::SUB) {
        return lhs - rhs;
    } else if constexpr (op == BinaryElementwiseOperator::MUL) {
        return lhs * rhs;
    } else if constexpr (op == BinaryElementwiseOperator::DIV) {
        return lhs / rhs;
    } else {
        KAI_ERROR("Unsupported binary element-wise operator!");
    }
}

/// Binary element-wise function.
///
/// @tparam op Binary element-wise operator to perform.
/// @tparam T Data type.
///
/// @param[in] lhs LHS data buffer.
/// @param[in] rhs RHS data buffer.
/// @param[in] lhs_height LHS height.
/// @param[in] lhs_width LHS width.
/// @param[in] rhs_height RHS height.
/// @param[in] rhs_width RHS width.
///
/// @return The result data buffer.
template <const BinaryElementwiseOperator op, typename T>
Buffer binary_elementwise_any_op_type(
    const void* lhs, const void* rhs, size_t lhs_height, size_t lhs_width, size_t rhs_height, size_t rhs_width) {
    const auto height = std::max(lhs_height, rhs_height);
    const auto width = std::max(lhs_width, rhs_width);

    KAI_ASSUME_ALWAYS(width * size_in_bits<T> % 8 == 0);
    Buffer dst(height * width * size_in_bits<T> / 8);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            const auto lhs_y = lhs_height > 1 ? y : 0;
            const auto lhs_x = lhs_width > 1 ? x : 0;
            const auto lhs_value = read_array<T>(lhs, lhs_y * lhs_width + lhs_x);

            const auto rhs_y = rhs_height > 1 ? y : 0;
            const auto rhs_x = rhs_width > 1 ? x : 0;
            const auto rhs_value = read_array<T>(rhs, rhs_y * rhs_width + rhs_x);

            const auto dst_value = scalar_binary_elementwise<op, T>(lhs_value, rhs_value);
            write_array<T>(dst.data(), y * width + x, dst_value);
        }
    }

    return dst;
}

template <const BinaryElementwiseOperator op>
Buffer binary_elementwise_any_type(
    const void* lhs, DataType lhs_dt, size_t lhs_height, size_t lhs_width,  //
    const void* rhs, DataType rhs_dt, size_t rhs_height, size_t rhs_width) {
    KAI_ASSUME_ALWAYS(lhs_dt == rhs_dt);
    KAI_ASSUME_ALWAYS(lhs_height == 1 || rhs_height == 1 || lhs_height == rhs_height);
    KAI_ASSUME_ALWAYS(lhs_width == 1 || rhs_width == 1 || lhs_width == rhs_width);

    switch (lhs_dt) {
        case DataType::FP32:
            return binary_elementwise_any_op_type<op, float>(lhs, rhs, lhs_height, lhs_width, rhs_height, rhs_width);

        case DataType::FP16:
            return binary_elementwise_any_op_type<op, Float16>(lhs, rhs, lhs_height, lhs_width, rhs_height, rhs_width);

        case DataType::I32:
            return binary_elementwise_any_op_type<op, int32_t>(lhs, rhs, lhs_height, lhs_width, rhs_height, rhs_width);

        case DataType::QSU4:
            return binary_elementwise_any_op_type<op, UInt4>(lhs, rhs, lhs_height, lhs_width, rhs_height, rhs_width);

        default:
            KAI_ERROR("Unsupported data type!");
    }
}

}  // namespace

Buffer add(
    const void* lhs, DataType lhs_dt, size_t lhs_height, size_t lhs_width,  //
    const void* rhs, DataType rhs_dt, size_t rhs_height, size_t rhs_width) {
    return binary_elementwise_any_type<BinaryElementwiseOperator::ADD>(
        lhs, lhs_dt, lhs_height, lhs_width, rhs, rhs_dt, rhs_height, rhs_width);
}

Buffer sub(
    const void* lhs, DataType lhs_dt, size_t lhs_height, size_t lhs_width,  //
    const void* rhs, DataType rhs_dt, size_t rhs_height, size_t rhs_width) {
    return binary_elementwise_any_type<BinaryElementwiseOperator::SUB>(
        lhs, lhs_dt, lhs_height, lhs_width, rhs, rhs_dt, rhs_height, rhs_width);
}

template <typename T>
Buffer sub(
    const void* lhs, size_t lhs_height, size_t lhs_width,  //
    const void* rhs, size_t rhs_height, size_t rhs_width) {
    return binary_elementwise_any_op_type<BinaryElementwiseOperator::SUB, T>(
        lhs, rhs, lhs_height, lhs_width, rhs_height, rhs_width);
}

template Buffer sub<int32_t>(
    const void* lhs, size_t lhs_height, size_t lhs_width,  //
    const void* rhs, size_t rhs_height, size_t rhs_width);

Buffer mul(
    const void* lhs, DataType lhs_dt, size_t lhs_height, size_t lhs_width,  //
    const void* rhs, DataType rhs_dt, size_t rhs_height, size_t rhs_width) {
    return binary_elementwise_any_type<BinaryElementwiseOperator::MUL>(
        lhs, lhs_dt, lhs_height, lhs_width, rhs, rhs_dt, rhs_height, rhs_width);
}

template <typename T>
Buffer mul(
    const void* lhs, size_t lhs_height, size_t lhs_width,  //
    const void* rhs, size_t rhs_height, size_t rhs_width) {
    return binary_elementwise_any_op_type<BinaryElementwiseOperator::MUL, T>(
        lhs, rhs, lhs_height, lhs_width, rhs_height, rhs_width);
}

template Buffer mul<float>(
    const void* lhs, size_t lhs_height, size_t lhs_width,  //
    const void* rhs, size_t rhs_height, size_t rhs_width);

template Buffer mul<int32_t>(
    const void* lhs, size_t lhs_height, size_t lhs_width,  //
    const void* rhs, size_t rhs_height, size_t rhs_width);

Buffer div(
    const void* lhs, DataType lhs_dt, size_t lhs_height, size_t lhs_width,  //
    const void* rhs, DataType rhs_dt, size_t rhs_height, size_t rhs_width) {
    return binary_elementwise_any_type<BinaryElementwiseOperator::DIV>(
        lhs, lhs_dt, lhs_height, lhs_width, rhs, rhs_dt, rhs_height, rhs_width);
}

}  // namespace kai::test
