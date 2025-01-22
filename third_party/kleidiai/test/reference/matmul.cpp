//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/reference/matmul.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "kai/kai_common.h"
#include "test/common/bfloat16.hpp"
#include "test/common/data_format.hpp"
#include "test/common/data_type.hpp"
#include "test/common/float16.hpp"
#include "test/common/int4.hpp"
#include "test/common/memory.hpp"
#include "test/common/round.hpp"
#include "test/reference/binary_elementwise.hpp"
#include "test/reference/cast.hpp"
#include "test/reference/pack.hpp"
#include "test/reference/reduce.hpp"
#include "test/reference/transpose.hpp"

namespace kai::test {

namespace {

/// Matrix multiplication.
///
/// @tparam T Data type.
///
/// @param[in] lhs LHS operand data buffer.
/// @param[in] rhs RHS operand data buffer.
/// @param[in] m Output height.
/// @param[in] n Output width.
/// @param[in] k Non-transposed LHS width and non-transposed RHS height.
/// @param[in] lhs_transposed `true` if LHS operand is transposed.
/// @param[in] rhs_transposed `true` if RHS operand is transposed.
///
/// @return The result data buffer.
template <typename T>
std::vector<uint8_t> matmul_any_type(
    const void* lhs, const void* rhs,  //
    size_t m, size_t n, size_t k,      //
    bool lhs_transposed, bool rhs_transposed) {
    const auto lhs_m_stride = lhs_transposed ? 1 : k;
    const auto lhs_k_stride = lhs_transposed ? m : 1;

    const auto rhs_n_stride = rhs_transposed ? k : 1;
    const auto rhs_k_stride = rhs_transposed ? 1 : n;

    std::vector<uint8_t> dst;
    dst.resize(m * n * size_in_bits<T> / 8);
    KAI_ASSUME(n * size_in_bits<T> % 8 == 0);

    for (size_t im = 0; im < m; ++im) {
        for (size_t in = 0; in < n; ++in) {
            T acc{0};

            for (size_t ik = 0; ik < k; ++ik) {
                const auto lhs_value = read_array<T>(lhs, im * lhs_m_stride + ik * lhs_k_stride);
                const auto rhs_value = read_array<T>(rhs, in * rhs_n_stride + ik * rhs_k_stride);
                acc += lhs_value * rhs_value;
            }

            write_array<T>(dst.data(), im * n + in, acc);
        }
    }

    return dst;
}

}  // namespace

std::vector<uint8_t> matmul_pack_rhs(
    const void* data, const void* scales, const void* zero_points, const DataFormat& src_format,
    const DataFormat& dst_format, size_t n, size_t k, bool transposing) {
    const auto src_dt = src_format.data_type();
    const auto src_pf = src_format.pack_format();

    const auto dst_dt = dst_format.data_type();
    const auto dst_pf = dst_format.pack_format();

    std::vector<uint8_t> tmp_data;
    std::vector<uint8_t> tmp_scales;
    std::vector<uint8_t> tmp_zero_points;

    if (transposing) {
        tmp_data = transpose(data, src_dt, k, n);
        data = tmp_data.data();
    }

    if (src_dt == DataType::QSU4 && src_pf == DataFormat::PackFormat::NONE &&  //
        dst_dt == DataType::QSI4 && dst_pf == DataFormat::PackFormat::QUANTIZE_PER_ROW) {
        // For this specific RHS format conversion:
        //
        //   * 4-bit data is added by 8.
        //   * Scale is divided by 16.
        //   * Zero point is accumulation of all values in the same row.

        KAI_ASSUME(zero_points == nullptr);
        const int32_t zero_point = 8;
        const uint8_t zero_point_i4 = UInt4::pack_u8(UInt4(zero_point), UInt4(zero_point));
        const int32_t row_zero_point = zero_point * static_cast<int32_t>(k);

        KAI_ASSUME(dst_format.subblock_width() > 0);
        const auto subblock_width_i32 = static_cast<int32_t>(dst_format.subblock_width());
        const auto subblock_width_f = static_cast<float>(dst_format.subblock_width());

        tmp_zero_points = reduce_add(data, src_format, n, k, DataFormat(DataType::I32), 0);
        tmp_zero_points = sub(tmp_zero_points.data(), DataType::I32, n, 1, &row_zero_point, DataType::I32, 1, 1);
        tmp_zero_points = mul(tmp_zero_points.data(), DataType::I32, n, 1, &subblock_width_i32, DataType::I32, 1, 1);
        zero_points = tmp_zero_points.data();

        tmp_data = add(data, DataType::QSU4, n, k, &zero_point_i4, DataType::QSU4, 1, 1);
        data = tmp_data.data();

        tmp_scales = div(scales, DataType::FP32, n, 1, &subblock_width_f, DataType::FP32, 1, 1);
        scales = tmp_scales.data();
    }

    return pack(dst_format, data, scales, zero_points, src_format, n, k);
}

std::vector<uint8_t> matmul(
    const void* lhs, [[maybe_unused]] const void* lhs_scales, [[maybe_unused]] const void* lhs_zero_points,
    DataType lhs_dt,  //
    const void* rhs, [[maybe_unused]] const void* rhs_scales, [[maybe_unused]] const void* rhs_zero_points,
    DataType rhs_dt,                                                                            //
    const void* bias, const void* bias_scales, const void* bias_zero_points, DataType bias_dt,  //
    DataType dst_dt,                                                                            //
    size_t m, size_t n, size_t k,                                                               //
    bool lhs_transposed, bool rhs_transposed) {
    const auto lhs_h = lhs_transposed ? k : m;
    const auto lhs_w = lhs_transposed ? m : k;

    const auto rhs_h = rhs_transposed ? n : k;
    const auto rhs_w = rhs_transposed ? k : n;

    std::vector<uint8_t> tmp_lhs;
    std::vector<uint8_t> tmp_rhs;
    std::vector<uint8_t> tmp_dst;
    std::vector<uint8_t> tmp_bias;

    if (lhs_dt != dst_dt) {
        tmp_lhs = cast(lhs, lhs_dt, dst_dt, lhs_h, lhs_w);
        lhs = tmp_lhs.data();
    }

    if (rhs_dt != dst_dt) {
        tmp_rhs = cast(rhs, rhs_dt, dst_dt, rhs_h, rhs_w);
        rhs = tmp_rhs.data();
    }

    switch (dst_dt) {
        case DataType::FP32:
            tmp_dst = matmul_any_type<float>(lhs, rhs, m, n, k, lhs_transposed, rhs_transposed);
            break;

        case DataType::FP16:
            tmp_dst = matmul_any_type<Float16>(lhs, rhs, m, n, k, lhs_transposed, rhs_transposed);
            break;

        default:
            KAI_ERROR("Unknown data type!");
    }

    if (bias != nullptr) {
        if (bias_dt != dst_dt) {
            tmp_bias = cast(bias, bias_dt, dst_dt, 1, n);
            bias = tmp_bias.data();
        }

        KAI_ASSUME(!data_type_is_quantized(bias_dt));
        KAI_ASSUME(bias_scales == nullptr);
        KAI_ASSUME(bias_zero_points == nullptr);

        tmp_dst = add(tmp_dst.data(), dst_dt, m, n, bias, bias_dt, 1, n);
    }

    return tmp_dst;
}

template <
    typename LhsData, typename LhsScale, typename LhsZeroPoint, typename RhsData, typename RhsScale,
    typename RhsZeroPoint, typename BiasData, typename BiasScale, typename BiasZeroPoint, typename DstData>
std::vector<uint8_t> matmul_nt_t_quantized(
    size_t m, size_t n, size_t k,  //
    const void* lhs_data, const void* lhs_scales, const void* lhs_zero_points, size_t lhs_quant_height,
    size_t lhs_quant_width,  //
    const void* rhs_data, const void* rhs_scales, const void* rhs_zero_points, size_t rhs_quant_height,
    size_t rhs_quant_width,  //
    const void* bias_data, const void* bias_scales, const void* bias_zero_points, size_t bias_quant_width) {
    const auto lhs_num_quant_per_row = round_up_division(k, lhs_quant_width);
    const auto rhs_num_quant_per_row = round_up_division(k, rhs_quant_width);

    std::vector<uint8_t> dst(m * n * sizeof(DstData));

    for (size_t y = 0; y < m; ++y) {
        for (size_t x = 0; x < n; ++x) {
            DstData acc = 0;

            for (size_t i = 0; i < k; ++i) {
                const auto lhs_data_index = y * k + i;
                const auto lhs_quant_index = y / lhs_quant_height * lhs_num_quant_per_row + i / lhs_quant_width;
                const auto lhs_value = read_array<LhsData>(lhs_data, lhs_data_index);
                const auto lhs_scale = lhs_scales != nullptr ? read_array<LhsScale>(lhs_scales, lhs_quant_index)
                                                             : static_cast<LhsScale>(1);
                const auto lhs_zero_point = lhs_zero_points != nullptr
                    ? read_array<LhsZeroPoint>(lhs_zero_points, lhs_quant_index)
                    : static_cast<LhsZeroPoint>(0);

                const auto rhs_data_index = x * k + i;
                const auto rhs_quant_index = x / rhs_quant_height * rhs_num_quant_per_row + i / rhs_quant_width;
                const auto rhs_value = read_array<RhsData>(rhs_data, rhs_data_index);
                const auto rhs_scale = rhs_scales != nullptr ? read_array<RhsScale>(rhs_scales, rhs_quant_index)
                                                             : static_cast<RhsScale>(1);
                const auto rhs_zero_point = rhs_zero_points != nullptr
                    ? read_array<RhsZeroPoint>(rhs_zero_points, rhs_quant_index)
                    : static_cast<RhsZeroPoint>(0);

                acc += (static_cast<DstData>(lhs_value) - static_cast<DstData>(lhs_zero_point)) *
                    static_cast<DstData>(lhs_scale) *
                    (static_cast<DstData>(rhs_value) - static_cast<DstData>(rhs_zero_point)) *
                    static_cast<DstData>(rhs_scale);
            }

            if (bias_data != nullptr) {
                const auto bias_value = read_array<BiasData>(bias_data, x);
                const auto bias_scale = bias_scales != nullptr
                    ? read_array<BiasScale>(bias_scales, x / bias_quant_width)
                    : static_cast<BiasScale>(1);
                const auto bias_zero_point = bias_zero_points != nullptr
                    ? read_array<BiasZeroPoint>(bias_zero_points, x / bias_quant_width)
                    : static_cast<BiasZeroPoint>(0);

                acc += (static_cast<DstData>(bias_value) - static_cast<DstData>(bias_zero_point)) *
                    static_cast<DstData>(bias_scale);
            }

            write_array<DstData>(dst.data(), y * n + x, acc);
        }
    }

    return dst;
}

template std::vector<uint8_t>
matmul_nt_t_quantized<int8_t, float, int32_t, int8_t, float, int32_t, int32_t, float, int32_t, float>(
    size_t m, size_t n, size_t k,  //
    const void* lhs_data, const void* lhs_scales, const void* lhs_zero_points, size_t lhs_quant_height,
    size_t lhs_quant_width,  //
    const void* rhs_data, const void* rhs_scales, const void* rhs_zero_points, size_t rhs_quant_height,
    size_t rhs_quant_width,  //
    const void* bias_data, const void* bias_scales, const void* bias_zero_points, size_t bias_quant_width);

template <
    typename LhsData, typename LhsScale, typename LhsZeroPoint, typename RhsData, typename RhsScale,
    typename RhsZeroPoint, typename Bias, typename IntAcc, typename DstData>
std::vector<uint8_t> matmul_clamp_nt_t(
    size_t m, size_t n, size_t k,                                                                       //
    const void* lhs_data, const void* lhs_scales, const void* lhs_zero_points, size_t lhs_quant_width,  //
    const void* rhs_data, const void* rhs_scales, const void* rhs_zero_points, size_t rhs_quant_width,  //
    const void* biases,                                                                                 //
    DstData min_value, DstData max_value) {
    const auto lhs_num_quant_per_row = round_up_division(k, lhs_quant_width);
    const auto rhs_num_quant_per_row = round_up_division(k, rhs_quant_width);

    std::vector<uint8_t> dst(m * n * sizeof(DstData));

    const auto* lhs_scales_ptr = reinterpret_cast<const LhsScale*>(lhs_scales);
    const auto* rhs_scales_ptr = reinterpret_cast<const RhsScale*>(rhs_scales);
    const auto* lhs_zero_points_ptr = reinterpret_cast<const LhsZeroPoint*>(lhs_zero_points);
    const auto* rhs_zero_points_ptr = reinterpret_cast<const RhsZeroPoint*>(rhs_zero_points);
    const auto* biases_ptr = reinterpret_cast<const Bias*>(biases);
    auto* dst_ptr = reinterpret_cast<DstData*>(dst.data());

    for (size_t y = 0; y < m; ++y) {
        for (size_t x = 0; x < n; ++x) {
            DstData acc = 0;

            for (size_t i = 0; i < k; ++i) {
                const auto lhs_value = read_array<LhsData>(lhs_data, y * k + i);
                const auto lhs_scale = lhs_scales_ptr[y * lhs_num_quant_per_row + i / lhs_quant_width];
                const auto lhs_zero_point = lhs_zero_points_ptr != nullptr
                    ? lhs_zero_points_ptr[y * lhs_num_quant_per_row + i / lhs_quant_width]
                    : 0;

                const auto rhs_value = read_array<RhsData>(rhs_data, x * k + i);
                const auto rhs_scale = rhs_scales_ptr[x * rhs_num_quant_per_row + i / rhs_quant_width];
                const auto rhs_zero_point = rhs_zero_points_ptr != nullptr
                    ? rhs_zero_points_ptr[y * rhs_num_quant_per_row + i / rhs_quant_width]
                    : 0;

                acc += static_cast<DstData>(
                           (static_cast<IntAcc>(lhs_value) - static_cast<IntAcc>(lhs_zero_point)) *
                           (static_cast<IntAcc>(rhs_value) - static_cast<IntAcc>(rhs_zero_point))) *
                    static_cast<DstData>(lhs_scale) * static_cast<DstData>(rhs_scale);
            }

            if (biases_ptr != nullptr) {
                acc += static_cast<DstData>(biases_ptr[x]);
            }

            acc = std::clamp(acc, min_value, max_value);
            dst_ptr[y * n + x] = acc;
        }
    }

    return dst;
}

template std::vector<uint8_t> matmul_clamp_nt_t<int8_t, float, int32_t, Int4, float, int32_t, float, int32_t, float>(
    size_t m, size_t n, size_t k,                                                                       //
    const void* lhs_data, const void* lhs_scales, const void* lhs_zero_points, size_t lhs_quant_width,  //
    const void* rhs_data, const void* rhs_scales, const void* rhs_zero_points, size_t rhs_quant_width,  //
    const void* biases,                                                                                 //
    float min_value, float max_value);

template std::vector<uint8_t>
matmul_clamp_nt_t<int8_t, Float16, int32_t, Int4, Float16, int32_t, float, int32_t, float>(
    size_t m, size_t n, size_t k,                                                                       //
    const void* lhs_data, const void* lhs_scales, const void* lhs_zero_points, size_t lhs_quant_width,  //
    const void* rhs_data, const void* rhs_scales, const void* rhs_zero_points, size_t rhs_quant_width,  //
    const void* biases,                                                                                 //
    float min_value, float max_value);

template std::vector<uint8_t> matmul_clamp_nt_t<int8_t, float, int32_t, Int4, BFloat16, int32_t, float, int32_t, float>(
    size_t m, size_t n, size_t k,                                                                       //
    const void* lhs_data, const void* lhs_scales, const void* lhs_zero_points, size_t lhs_quant_width,  //
    const void* rhs_data, const void* rhs_scales, const void* rhs_zero_points, size_t rhs_quant_width,  //
    const void* biases,                                                                                 //
    float min_value, float max_value);

template std::vector<uint8_t> matmul_clamp_nt_t<int8_t, float, int32_t, int8_t, float, int32_t, float, int32_t, float>(
    size_t m, size_t n, size_t k,                                                                       //
    const void* lhs_data, const void* lhs_scales, const void* lhs_zero_points, size_t lhs_quant_width,  //
    const void* rhs_data, const void* rhs_scales, const void* rhs_zero_points, size_t rhs_quant_width,  //
    const void* biases,                                                                                 //
    float min_value, float max_value);

template <
    typename LhsData, typename LhsScale, typename LhsZeroPoint, typename RhsData, typename RhsScale,
    typename RhsZeroPoint, typename Bias, typename IntAcc, typename DstData>
std::vector<uint8_t> matmul_clamp_nt_nt(
    size_t m, size_t n, size_t k,                                                                       //
    const void* lhs_data, const void* lhs_scales, const void* lhs_zero_points, size_t lhs_quant_width,  //
    const void* rhs_data, const void* rhs_scales, const void* rhs_zero_points, size_t rhs_quant_width,  //
    const void* biases,                                                                                 //
    DstData min_value, DstData max_value) {
    const auto lhs_num_quant_per_row = round_up_division(k, lhs_quant_width);
    const auto rhs_num_quant_per_row = round_up_division(k, rhs_quant_width);

    std::vector<uint8_t> dst(m * n * sizeof(DstData));

    const auto* lhs_scales_ptr = reinterpret_cast<const LhsScale*>(lhs_scales);
    const auto* rhs_scales_ptr = reinterpret_cast<const RhsScale*>(rhs_scales);
    const auto* lhs_zero_points_ptr = reinterpret_cast<const LhsZeroPoint*>(lhs_zero_points);
    const auto* rhs_zero_points_ptr = reinterpret_cast<const RhsZeroPoint*>(rhs_zero_points);
    const auto* biases_ptr = reinterpret_cast<const Bias*>(biases);
    auto* dst_ptr = reinterpret_cast<DstData*>(dst.data());

    for (size_t y = 0; y < m; ++y) {
        for (size_t x = 0; x < n; ++x) {
            DstData acc = 0;

            for (size_t i = 0; i < k; ++i) {
                const auto lhs_value = read_array<LhsData>(lhs_data, y * k + i);
                const auto lhs_scale = lhs_scales_ptr[y * lhs_num_quant_per_row + i / lhs_quant_width];
                const auto lhs_zero_point = lhs_zero_points_ptr != nullptr
                    ? lhs_zero_points_ptr[y * lhs_num_quant_per_row + i / lhs_quant_width]
                    : 0;

                const auto rhs_value = read_array<RhsData>(rhs_data, x + i * n);
                const auto rhs_scale = rhs_scales_ptr[x * rhs_num_quant_per_row + i / rhs_quant_width];
                const auto rhs_zero_point = rhs_zero_points_ptr != nullptr
                    ? rhs_zero_points_ptr[y * rhs_num_quant_per_row + i / rhs_quant_width]
                    : 0;

                acc += static_cast<DstData>(
                           (static_cast<IntAcc>(lhs_value) - static_cast<IntAcc>(lhs_zero_point)) *
                           (static_cast<IntAcc>(rhs_value) - static_cast<IntAcc>(rhs_zero_point))) *
                    static_cast<DstData>(lhs_scale) * static_cast<DstData>(rhs_scale);
            }

            if (biases_ptr != nullptr) {
                acc += static_cast<DstData>(biases_ptr[x]);
            }

            acc = std::clamp(acc, min_value, max_value);
            dst_ptr[y * n + x] = acc;
        }
    }

    return dst;
}

template std::vector<uint8_t> matmul_clamp_nt_nt<int8_t, float, int32_t, Int4, float, int32_t, float, int32_t, float>(
    size_t m, size_t n, size_t k,                                                                       //
    const void* lhs_data, const void* lhs_scales, const void* lhs_zero_points, size_t lhs_quant_width,  //
    const void* rhs_data, const void* rhs_scales, const void* rhs_zero_points, size_t rhs_quant_width,  //
    const void* biases,                                                                                 //
    float min_value, float max_value);

template std::vector<uint8_t>
matmul_clamp_nt_nt<int8_t, Float16, int32_t, Int4, Float16, int32_t, float, int32_t, float>(
    size_t m, size_t n, size_t k,                                                                       //
    const void* lhs_data, const void* lhs_scales, const void* lhs_zero_points, size_t lhs_quant_width,  //
    const void* rhs_data, const void* rhs_scales, const void* rhs_zero_points, size_t rhs_quant_width,  //
    const void* biases,                                                                                 //
    float min_value, float max_value);

template std::vector<uint8_t>
matmul_clamp_nt_nt<int8_t, float, int32_t, Int4, BFloat16, int32_t, float, int32_t, float>(
    size_t m, size_t n, size_t k,                                                                       //
    const void* lhs_data, const void* lhs_scales, const void* lhs_zero_points, size_t lhs_quant_width,  //
    const void* rhs_data, const void* rhs_scales, const void* rhs_zero_points, size_t rhs_quant_width,  //
    const void* biases,                                                                                 //
    float min_value, float max_value);

}  // namespace kai::test
