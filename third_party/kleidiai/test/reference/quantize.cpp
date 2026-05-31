//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/reference/quantize.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <tuple>

#include "test/common/bfloat16.hpp"
#include "test/common/buffer.hpp"
#include "test/common/int4.hpp"
#include "test/common/memory.hpp"
#include "test/common/numeric_limits.hpp"
#include "test/common/round.hpp"
#include "test/common/type_traits.hpp"
#include "test/reference/cast.hpp"

namespace kai::test {

namespace {

template <typename FloatData, typename IntData, typename ZeroPoint>
std::tuple<FloatData, ZeroPoint> get_scale_zero_point_from_range(FloatData min_value, FloatData max_value) {
    const FloatData q_min = numeric_lowest<IntData>;
    const FloatData q_max = numeric_highest<IntData>;

    if (min_value > 0) {
        min_value = 0;
    }

    if (max_value < 0) {
        max_value = 0;
    }

    // The reason for computing the inverted scale first is to make it bit-perfect with quantized packing
    // micro-kernels. If those micro-kernels don't do it this way anymore, it makes more sense to calculate
    // the scale directly.
    const FloatData inv_scale = max_value != min_value ? (q_max - q_min) / (max_value - min_value) : 1.0F;
    const FloatData scale = 1.0F / inv_scale;

    const FloatData scaled_min = min_value / scale;
    const FloatData scaled_max = max_value / scale;

    const FloatData zero_point_f = -(scaled_min + q_min) < scaled_max + q_max ? scaled_min - q_min : scaled_max - q_max;
    const ZeroPoint zero_point = -round_to_nearest_even<ZeroPoint>(zero_point_f);

    return {scale, zero_point};
}

/// Quantized a float value to an integer datatype using a provided scale.
///
/// @tparam IntType Quantized integer datatype.
///
/// @param[in] float The value to quantize
/// @param[in] scale The scale used to quantize the provided float value.
///
/// @return The quantized data matrix, the quantization scale matrix and the quantization zero point matrix.
template <typename IntType>
IntType quantize_symmetric(float value, float scale) {
    const auto inv_scale = scale != 0 ? 1.0F / scale : 0.0F;
    auto qsi32 = round_to_nearest_even_i32(value * inv_scale);

    if (is_unsigned<IntType>) {
        qsi32 += 1 << (size_in_bits<IntType> - 1);
    }

    return static_cast<IntType>(std::clamp<int32_t>(qsi32, numeric_lowest<IntType>, numeric_highest<IntType>));
}

/// Computes the quantization information using symmetric per-block quantization method.
///
/// The input matrix is divided into quantization blocks of the same size.
///
/// The height of the block does not affect the behavior of this function hence it is omitted
/// from the function arguments and the figures below.
///
/// ```
/// Quantization blocks -------+
///          |                 |
///          |                 |
///          v                 v
/// +-----------------+-----------------+----- ...
/// | f00 f01 f02 f03 | f04 f05 f06 f07 | ........
/// | f10 f11 f12 f13 | f14 f15 f16 f17 | ........
/// | f20 f21 f22 f23 | f24 f25 f26 f27 | ........
/// | f30 f31 f32 f33 | f34 f35 f36 f37 | ........
/// | ............... | ............... | ........
/// : ............... : ............... : ........
/// ```
///
/// Each row of the quantization block is quantized individually.
///
/// ```
/// Floating-point data           Scale
/// +-----------------+          +-----+
/// | f00 f01 f02 f03 | -------> | s00 |
/// | f10 f11 f12 f13 | -------> | s10 |
/// | f20 f21 f22 f23 | -------> | s20 |
/// | f30 f31 f32 f33 | -------> | s30 |
/// | ............... |          | ... |
/// : ............... :          : ... :
/// ```
///
/// The computed quantization scale matrix:
///
/// ```
/// +-----+-----+-- ...
/// | s00 | s01 | .....
/// | s10 | s11 | .....
/// | s20 | s21 | .....
/// | s30 | s31 | .....
/// | ... | ... | .....
/// : ... : ... : .....
/// ```
///
/// @tparam SrcType The data type of the input data (must be floating-point).
/// @tparam DstType The data type of the output data (must be integer).
/// @tparam ScaleType The data type of the quantization scales (must be floating-point).
///
/// @param[in] src The input matrix.
/// @param[in] height The number of rows.
/// @param[in] width The number of columns.
/// @param[in] quant_width The number of columns of the quantization block.
///
/// @return The quantization scale matrix.
template <typename SrcType, typename DstType, typename ScaleType>
Buffer compute_symmetric_per_block_quantization_info(const void* src, size_t height, size_t width, size_t quant_width) {
    static_assert(is_floating_point<SrcType>);
    static_assert(is_integral<DstType>);
    static_assert(is_floating_point<ScaleType>);

    KAI_ASSUME_ALWAYS(quant_width != 0);

    const auto num_quant_packets_x = round_up_division(width, quant_width);

    const auto scales_bytes = height * num_quant_packets_x * sizeof(ScaleType);
    Buffer scales(scales_bytes);

    const auto* src_ptr = reinterpret_cast<const SrcType*>(src);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x_quant = 0; x_quant < width; x_quant += quant_width) {
            // Computes the quantization scale.
            SrcType max_abs = 0;

            for (size_t x_element = 0; x_element < quant_width; ++x_element) {
                const auto x = x_quant + x_element;

                if (x < width) {
                    max_abs = std::max<SrcType>(max_abs, std::abs(src_ptr[y * width + x]));
                }
            }

            const auto scale =
                max_abs / static_cast<SrcType>((static_cast<uint64_t>(1) << (size_in_bits<DstType> - 1)) - 1);

            // Stores the scales.
            write_array<ScaleType>(scales.data(), y * num_quant_packets_x + x_quant / quant_width, scale);
        }
    }

    return scales;
}

/// Dynamically quantizes each block of the matrix using symmetric quantization method.
///
/// The quantization information is calculated using
/// @ref compute_symmetric_per_block_quantization_info function.
/// The floating-point data is then quantized using
/// @ref quantize_symmetric_per_block function.
///
/// To retain highest quantization accuracy, the data is quantized using the quantization scale
/// with the same data type as the input data.
/// After that the quantization scale can be stored in the buffer using `ScaleType` data type
/// which might have lowest precision than the input data type.
///
/// @tparam SrcType The data type of the input data (must be floating-point).
/// @tparam DstType The data type of the output data (must be integer).
/// @tparam ScaleType The data type of the quantization scales (must be floating-point).
///
/// @param[in] src The input matrix.
/// @param[in] height The number of rows.
/// @param[in] width The number of columns.
/// @param[in] quant_width The number of columns of the quantization block.
///
/// @return The quantized data matrix and the quantization scale matrix.
template <typename SrcType, typename DstType, typename ScaleType>
std::tuple<Buffer, Buffer> quantize_symmetric_per_block_dynamic(
    const void* src, size_t height, size_t width, size_t quant_width) {
    auto scales_src_type =
        compute_symmetric_per_block_quantization_info<SrcType, DstType, SrcType>(src, height, width, quant_width);
    auto data = quantize_symmetric_per_block<SrcType, DstType, SrcType>(
        src, scales_src_type.data(), height, width, quant_width);

    if constexpr (std::is_same_v<ScaleType, SrcType>) {
        return {std::move(data), std::move(scales_src_type)};
    } else {
        auto scales =
            cast<ScaleType, SrcType>(scales_src_type.data(), scales_src_type.size() * 8 / size_in_bits<SrcType>);

        return {std::move(data), std::move(scales)};
    }
}

/// Dynamically quantizes each block of the matrix using symmetric quantization method.
///
/// @param[in] src The input matrix.
/// @param[in] src_type The data type of the input data (must be FP32).
/// @param[in] height The number of rows.
/// @param[in] width The number of columns.
/// @param[in] qinfo The quantization information.
///
/// @return The quantized data matrix and the quantization scale matrix.
std::tuple<Buffer, Buffer> quantize_symmetric_per_block_dynamic(
    const void* src, DataType src_type, size_t height, size_t width, const QuantizationInfo& qinfo) {
    // Fail fast for datatypes that must be fixed.
    KAI_ASSUME_ALWAYS(src_type == DataType::FP32);

    switch (qinfo.dst_type) {
        case DataType::QSI4:
            switch (qinfo.scale_type) {
                case DataType::FP16:
                    return quantize_symmetric_per_block_dynamic<float, Int4, Float16>(
                        src, height, width, qinfo.quant_width);
                case DataType::FP32:
                    return quantize_symmetric_per_block_dynamic<float, Int4, float>(
                        src, height, width, qinfo.quant_width);
                case DataType::BF16:
                    return quantize_symmetric_per_block_dynamic<float, Int4, BFloat16<>>(
                        src, height, width, qinfo.quant_width);
                default:
                    break;
            }
            break;
        case DataType::QSI8:
            switch (qinfo.scale_type) {
                case DataType::FP16:
                    return quantize_symmetric_per_block_dynamic<float, int8_t, Float16>(
                        src, height, width, qinfo.quant_width);
                case DataType::FP32:
                    return quantize_symmetric_per_block_dynamic<float, int8_t, float>(
                        src, height, width, qinfo.quant_width);
                default:
                    break;
            }
            break;
        case DataType::I32:
            if (qinfo.scale_type == DataType::FP32) {
                return quantize_symmetric_per_block_dynamic<float, int32_t, float>(
                    src, height, width, qinfo.quant_width);
            }
            break;
        default:
            break;
    }
    KAI_ERROR("Unsupported combination of data types for symmetric quantization.");
}

/// Dynamically quantizes each block of the matrix using asymmetric quantization method.
///
/// The quantization information is calculated using
/// @ref compute_asymmetric_per_block_quantization_info function.
/// The floating-point data is then quantized using
/// @ref quantize_asymmetric_per_block function.
///
/// To retain highest quantization accuracy, the data is quantized using the quantization scale
/// with the same data type as the input data.
/// After that the quantization scale can be stored in the buffer using `ScaleType` data type
/// which might have lowest precision than the input data type.
///
/// @tparam SrcType The data type of the input data (must be floating-point).
/// @tparam DstType The data type of the output data (must be integer).
/// @tparam ScaleType The data type of the quantization scales (must be floating-point).
/// @tparam ZeroPointType The data type of the quantization zero points (must be integer).
///
/// @param[in] src The input matrix.
/// @param[in] height The number of rows.
/// @param[in] width The number of columns.
/// @param[in] quant_width The number of columns of the quantization block.
///
/// @return The quantized data matrix, the quantization scale matrix and the quantization zero point matrix.
template <typename SrcType, typename DstType, typename ScaleType, typename ZeroPointType>
std::tuple<Buffer, Buffer, Buffer> quantize_asymmetric_per_block_dynamic(
    const void* src, size_t height, size_t width, size_t quant_width) {
    /* Calculate the asymmetric quantization information, one scaling per row  */
    auto [scales_src_type, zero_points] =
        compute_asymmetric_per_block_quantization_info<SrcType, DstType, SrcType, ZeroPointType>(
            src, height, width, quant_width);

    /* Do the actual quantization */
    auto data = quantize_asymmetric_per_block<SrcType, DstType, SrcType, ZeroPointType>(
        src, scales_src_type.data(), zero_points.data(), height, width, quant_width);

    if constexpr (std::is_same_v<ScaleType, SrcType>) {
        return {std::move(data), std::move(scales_src_type), std::move(zero_points)};
    } else {
        auto scales =
            cast<ScaleType, SrcType>(scales_src_type.data(), scales_src_type.size() * 8 / size_in_bits<SrcType>);

        return {std::move(data), std::move(scales), std::move(zero_points)};
    }
}

/// Dynamically quantizes each block of the matrix using asymmetric quantization method.
///
/// @param[in] src The input matrix.
/// @param[in] src_type The data type of the input data (must be FP32).
/// @param[in] height The number of rows.
/// @param[in] width The number of columns.
/// @param[in] qinfo The quantization information.
///
/// @return The quantized data matrix, the quantization scale matrix and the quantization zero point matrix.
std::tuple<Buffer, Buffer, Buffer> quantize_asymmetric_per_block_dynamic(
    const void* src, DataType src_type, size_t height, size_t width, const QuantizationInfo& qinfo) {
    // Fail fast for datatypes that must be fixed.
    KAI_ASSUME_ALWAYS(src_type == DataType::FP32);
    KAI_ASSUME_ALWAYS(qinfo.zero_point_type == DataType::I32);

    switch (qinfo.dst_type) {
        case DataType::QAI8:
            switch (qinfo.scale_type) {
                case DataType::FP32:
                    return quantize_asymmetric_per_block_dynamic<float, int8_t, float, int32_t>(
                        src, height, width, qinfo.quant_width);
                case DataType::BF16:
                    return quantize_asymmetric_per_block_dynamic<float, int8_t, BFloat16<>, int32_t>(
                        src, height, width, qinfo.quant_width);
                default:
                    break;
            }
            break;
        case DataType::QAI4:
            switch (qinfo.scale_type) {
                case DataType::FP32:
                    return quantize_asymmetric_per_block_dynamic<float, Int4, float, int32_t>(
                        src, height, width, qinfo.quant_width);
                default:
                    break;
            }
            break;
        default:
            break;
    }
    KAI_ERROR("Unsupported combination of destination/scale types for asymmetric quantization.");
}

}  // namespace

template <typename FloatType, typename IntType, typename ZeroPointType>
IntType quantize_asymmetric(FloatType value, FloatType scale, ZeroPointType zero_point) {
    const auto inv_scale = scale != 0 ? 1.0F / scale : 0.0F;
    auto quantized_value = round_to_nearest_even<ZeroPointType>(value * inv_scale) + zero_point;
    return static_cast<IntType>(
        std::clamp<ZeroPointType>(quantized_value, numeric_lowest<IntType>, numeric_highest<IntType>));
}

template int8_t quantize_asymmetric(float value, float scale, int32_t zero_point);

template <typename SrcType, typename DstType, typename ScaleType>
Buffer quantize_symmetric_per_block(
    const void* src, const void* scales, size_t height, size_t width, size_t quant_width) {
    static_assert(is_floating_point<SrcType>);
    static_assert(is_integral<DstType>);
    static_assert(is_floating_point<ScaleType>);

    const auto num_quant_packets_x = round_up_division(width, quant_width);

    const auto data_bytes = round_up_division(height * width * size_in_bits<DstType>, 8);
    Buffer data(data_bytes);

    const auto* src_ptr = reinterpret_cast<const SrcType*>(src);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x_quant = 0; x_quant < width; x_quant += quant_width) {
            const auto scale = read_array<ScaleType>(scales, y * num_quant_packets_x + x_quant / quant_width);

            // Quantizes and stores the data.
            for (size_t x_element = 0; x_element < quant_width; ++x_element) {
                const auto x = x_quant + x_element;

                if (x < width) {
                    const auto quantized = quantize_symmetric<DstType>(src_ptr[y * width + x], scale);
                    write_array(data.data(), y * width + x, quantized);
                }
            }
        }
    }
    return data;
}

template Buffer quantize_symmetric_per_block<float, int32_t, float>(
    const void* src, const void* scales, size_t height, size_t width, size_t quant_width);

template <typename SrcType, typename DstType, typename ScaleType, typename ZeroPointType>
std::tuple<Buffer, Buffer> compute_asymmetric_per_block_quantization_info(
    const void* src, size_t height, size_t width, size_t quant_width) {
    static_assert(is_floating_point<SrcType>);
    static_assert(is_integral<DstType>);
    static_assert(is_floating_point<ScaleType>);
    static_assert(is_integral<ZeroPointType>);

    KAI_ASSUME_ALWAYS(quant_width != 0);

    const auto num_quant_packets_x = round_up_division(width, quant_width);

    const auto scales_bytes = height * num_quant_packets_x * sizeof(ScaleType);
    Buffer scales(scales_bytes);

    const auto zero_points_bytes = height * num_quant_packets_x * sizeof(ZeroPointType);
    Buffer zero_points(zero_points_bytes);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x_quant = 0; x_quant < width; x_quant += quant_width) {
            // Computes the quantization scale and zero point.
            auto min_value = numeric_highest<SrcType>;
            auto max_value = numeric_lowest<SrcType>;

            for (size_t x_element = 0; x_element < quant_width; ++x_element) {
                const auto x = x_quant + x_element;

                if (x < width) {
                    const auto value = read_array<SrcType>(src, y * width + x);

                    min_value = std::min(min_value, value);
                    max_value = std::max(max_value, value);
                }
            }

            const auto [scale, zero_point] =
                get_scale_zero_point_from_range<SrcType, DstType, ZeroPointType>(min_value, max_value);

            // Stores the scale and zero point.
            write_array<ScaleType>(scales.data(), y * num_quant_packets_x + x_quant / quant_width, scale);
            write_array<ZeroPointType>(zero_points.data(), y * num_quant_packets_x + x_quant / quant_width, zero_point);
        }
    }

    return {std::move(scales), std::move(zero_points)};
}

template <typename SrcType, typename DstType, typename ScaleType, typename ZeroPointType>
Buffer quantize_asymmetric_per_block(
    const void* src, const void* scales, const void* zero_points, size_t height, size_t width, size_t quant_width) {
    static_assert(is_floating_point<SrcType>);
    static_assert(is_integral<DstType>);
    static_assert(is_floating_point<ScaleType>);
    static_assert(is_integral<ZeroPointType>);

    const auto num_quant_packets_x = round_up_division(width, quant_width);

    const auto data_bytes = round_up_division(height * width * size_in_bits<DstType>, 8);
    Buffer data(data_bytes);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x_quant = 0; x_quant < width; x_quant += quant_width) {
            const auto scale = read_array<ScaleType>(scales, y * num_quant_packets_x + x_quant / quant_width);
            const auto zero_point =
                read_array<ZeroPointType>(zero_points, y * num_quant_packets_x + x_quant / quant_width);

            // Quantizes and stores the data.
            for (size_t x_element = 0; x_element < quant_width; ++x_element) {
                const auto x = x_quant + x_element;

                if (x < width) {
                    const auto value_f = read_array<SrcType>(src, y * width + x);
                    const auto value_q =
                        quantize_asymmetric<SrcType, DstType, ZeroPointType>(value_f, scale, zero_point);

                    write_array<DstType>(data.data(), y * width + x, value_q);
                }
            }
        }
    }

    return data;
}

std::tuple<Buffer, QuantizationOutputs> quantize_dynamic(
    const void* src, DataType src_type, size_t height, size_t width, const QuantizationInfo& qinfo) {
    KAI_ASSUME_ALWAYS(data_type_is_quantized(qinfo.dst_type));
    Buffer data;
    QuantizationOutputs qoutputs;
    if (data_type_is_quantized_asymm(qinfo.dst_type)) {
        KAI_ASSUME_ALWAYS(qinfo.zero_point_type != DataType::UNKNOWN);
        std::tie(data, qoutputs.scales, qoutputs.zero_points) =
            quantize_asymmetric_per_block_dynamic(src, src_type, height, width, qinfo);
    } else {
        std::tie(data, qoutputs.scales) = quantize_symmetric_per_block_dynamic(src, src_type, height, width, qinfo);
    }
    return {std::move(data), std::move(qoutputs)};
}
}  // namespace kai::test
