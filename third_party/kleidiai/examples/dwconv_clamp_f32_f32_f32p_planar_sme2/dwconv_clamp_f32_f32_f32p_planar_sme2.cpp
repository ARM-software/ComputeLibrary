//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <float.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <vector>

#include "kai/kai_common.h"
#include "kai/ukernels/dwconv/dwconv_f32_f32_f32p/kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla.h"
#include "kai/ukernels/dwconv/pack/kai_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme.h"

using VEC_TYPE = std::vector<float>;

namespace {
constexpr float clamp_min = std::numeric_limits<float>::lowest();
constexpr float clamp_max = std::numeric_limits<float>::max();

struct Padding2D {
    size_t left = 0;
    size_t right = 0;
    size_t bottom = 0;
    size_t top = 0;
};

struct Shape {
    size_t n = 1;
    size_t h = 1;
    size_t w = 1;
    size_t c = 1;

    [[nodiscard]] auto size() const -> size_t {
        return n * h * w * c;
    }

    friend std::ostream& operator<<(std::ostream& os, const Shape& shape) {
        os << " [ " << shape.n << " , " << shape.h << " ," << shape.w << " , " << shape.c << " ] ";
        return os;
    }

    constexpr const std::size_t& operator[](std::size_t idx) const {
        switch (idx) {
            case 0:
                return n;
            case 1:
                return h;
            case 2:
                return w;
            case 3:
                return c;
            default:
                throw std::out_of_range("Shape-index out of range (0-3)");
        }
    }
};

#ifdef KAI_DEBUG
void print_tensor(const Shape& shape, const char* name, const float* src) {
    std::cout << "\n\n" << name << " = [\n";
    for (size_t n = 0; n < shape.n; n++) {
        std::cout << "\n";
        for (size_t y = 0; y < shape.h; ++y) {
            std::cout << "  [";
            for (size_t x = 0; x < shape.w; x++) {
                std::cout << "[";
                for (size_t c = 0; c < shape.c; c++) {
                    if (c != 0) std::cout << " , ";
                    std::cout << std::setprecision(3) << std::fixed
                              << src[n * shape.h * shape.w * shape.c + y * shape.w * shape.c + x * shape.c + c];
                }
                std::cout << "] ";
            }
            std::cout << ("],\n");
        }
    }
    std::cout << ("]\n\n");
}

void print_raw(const Shape& shape, const char* name, const VEC_TYPE& src) {
    std::cout << "\n\n" << name << " = [";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i != 0) std::cout << " , ";
        std::cout << std::setprecision(1) << std::fixed << (float)src[i];
    }
    std::cout << "]\n";
}
#endif  // KAI_DEBUG

/// Fills the matrix with incremental values according to the provided weight.
/// @param[in] size Total number of elements to fill in passed vector;.
/// @param[in] dst Vector representing a tensor to fill.
/// @param[in] weight A weight value to increment by.
void fill_matrix(size_t size, VEC_TYPE& dst, const float weight) {
    for (size_t i = 0; i < size; i++) {
        dst[i] = float((10 * i) * weight);
    }
}

/// Depthwise Convolution - Expects NHWC dataformat. Padding value is 0.
///
/// @tparam T Data type.
///
/// @param[in] batches   Batch dimension of feature map.
/// @param[in] in_height height of feature map.
/// @param[in] in_width  width of feature map.
/// @param[in] channels  Number of channels in feature map.
/// @param[in] filter_height Height dimension in filter.
/// @param[in] filter_width  Width of convolution filter.
/// @param[in] feature_map Ptr to start of feature map.
/// @param[in] weights Ptr to start of weights buffer/tensor.
/// @param[in] bias Ptr to start of bias buffer.
/// @param[out] out Ptr to start of output buffer
/// @param[in] clamp_min float value to clamp output to (lower bound).
/// @param[in] clamp_max float value to clamp output to (upper bound).
/// @param[in] pad Struct describing padding dimensions
template <typename T>
void depthwise_reference(
    const size_t batches, const size_t in_height, const size_t in_width, const size_t channels,
    const size_t filter_height, const size_t filter_width, const void* feature_map, const void* weights,
    const void* bias, void* out, float clamp_min, float clamp_max, const Padding2D pad) {
    // Calculate output dims (Padding = Valid).
    const size_t out_height = (in_height + pad.top + pad.bottom + 1 - filter_height);
    const size_t out_width = in_width + pad.left + pad.right + 1 - filter_width;
    const size_t out_size = out_height * out_width * batches * channels;

    // We accumulate in FP32 and clamp and cast to return type later.
    std::vector<float> acc(out_size, 0.0f);

    for (size_t b = 0; b < batches; ++b) {
        for (size_t out_h = 0; out_h < out_height; ++out_h) {
            for (size_t out_w = 0; out_w < out_width; ++out_w) {
                const size_t out_base = ((b * out_height + out_h) * out_width + out_w) * channels;

                // Apply filter to feature map.
                for (size_t ic = 0; ic < channels; ++ic) {
                    float sum = 0.0f;

                    for (size_t kernel_h = 0; kernel_h < filter_height; ++kernel_h) {
                        // Determine if input height bounds. If not, then this is padding.
                        const int in_y = static_cast<int>(out_h + kernel_h) - static_cast<int>(pad.top);
                        if (in_y < 0 || in_height <= static_cast<size_t>(in_y)) continue;

                        for (size_t kernel_w = 0; kernel_w < filter_width; ++kernel_w) {
                            // Determine if in input width bounds, if not this is padding.
                            const int in_x = static_cast<int>(out_w + kernel_w) - static_cast<int>(pad.left);
                            if (in_x < 0 || in_width <= static_cast<size_t>(in_x)) continue;

                            auto in_idx = ((b * in_height + in_y) * in_width + in_x) * channels + ic;
                            auto weights_idx = ((kernel_h * filter_width) + kernel_w) * channels + ic;

                            auto wei_value = reinterpret_cast<const float*>(weights)[weights_idx];
                            auto in_value = reinterpret_cast<const float*>(feature_map)[in_idx];

                            // Perform actual accumulation and store in output vector
                            sum += in_value * wei_value;
                        }
                    }

                    auto out_idx = out_base + ic;
                    float bias_value = reinterpret_cast<const float*>(bias)[ic];
                    sum = sum + bias_value;
                    sum = std::clamp(sum, clamp_min, clamp_max);
                    reinterpret_cast<float*>(out)[out_idx] = sum;
                }
            }
        }
    }
}
}  // namespace

int main() {
    const int batches = 1;
    enum class pad_mode { SAME, VALID };

    size_t total_test = 0;
    for (pad_mode pad : {pad_mode::SAME, pad_mode::VALID}) {
        for (size_t width = 128; width < 129; width += 2) {
            for (size_t height = 141; height < 142; height += 2) {
                for (size_t channels = 1; channels < 64; channels += 7) {
                    total_test++;
                    const int filter_height = 3;
                    const int filter_width = 3;
                    const int depth_multiplier = 1;  // Only dm =1 supported.

                    assert(filter_height > 1 && filter_width > 1);

                    const size_t pad_total_height = (pad == pad_mode::SAME) ? filter_height - 1 : 0;
                    const size_t pad_total_width = (pad == pad_mode::SAME) ? filter_width - 1 : 0;
                    Padding2D padding;
                    padding.top = pad_total_height / 2;
                    padding.left = pad_total_width / 2;
                    padding.right = pad_total_width - padding.left;
                    padding.bottom = pad_total_height - padding.top;

                    Shape in_shape{batches, height, width, channels};
                    Shape wei_shape{filter_height, filter_width, channels, depth_multiplier};
                    Shape bias_shape{depth_multiplier * channels};
                    Shape out_shape{
                        batches, (height + padding.top + padding.bottom + 1 - filter_height),
                        (width + padding.left + padding.right + 1 - filter_width), channels * depth_multiplier};

                    VEC_TYPE input(in_shape.size(), 0.0f);
                    VEC_TYPE weights(wei_shape.size(), 0.1f);
                    VEC_TYPE bias(bias_shape.size(), 0.0f);
                    VEC_TYPE out(out_shape.size(), 0.0f);
                    VEC_TYPE ref(out_shape.size(), 0.0f);

                    fill_matrix(in_shape.size(), input, 0.01f);
                    fill_matrix(wei_shape.size(), weights, 0.02f);
                    fill_matrix(bias_shape.size(), bias, 1.f);

                    // For testing using Python.
#ifdef KAI_DEBUG
                    {
                        std::cout << "\n#BEGIN PARAMS\n";
                        std::cout << "\nbatch, height, width, channels = " << batches << ", " << height << ", " << width
                                  << ", " << channels << std::endl;
                        std::cout << "\nfilter_height, filter_width = " << filter_height << ", " << filter_width
                                  << std::endl;
                        print_raw(in_shape, "Inputs ", input);
                        print_raw(wei_shape, "Weights ", weights);
                        print_raw(bias_shape, "Bias ", bias);
                        std::cout << "\npad_top, pad_bottom  = " << padding.top << ", " << padding.bottom << std::endl;
                        std::cout << "\npad_left, pad_right  = " << padding.left << ", " << padding.right << std::endl
                                  << std::endl;
                        std::cout << "\n#END PARAMS\n";
                    }
#endif  // KAI_DEBUG

                    // -------------------------------------------------
                    // 1. Calculate Reference Depthwise Values.
                    // -------------------------------------------------
                    depthwise_reference<float>(
                        batches, height, width, channels, filter_height, filter_width, (const void*)input.data(),
                        (const void*)weights.data(), (const void*)bias.data(), (void*)ref.data(), clamp_min, clamp_max,
                        padding);

                    // -------------------------------------------------
                    // 2. Pack weights for use in SME Kernel
                    // -------------------------------------------------
                    const size_t packed_size =
                        kai_rhs_get_dst_size_dwconv_pack_x32p1vlx1b_x32_x32_sme(filter_height, filter_width, channels) /
                        sizeof(float);

                    // Run packing micro-kernel.
                    std::vector<float> weights_packed(packed_size);
                    kai_run_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme(
                        filter_height, filter_width, wei_shape[0], wei_shape[1], channels, weights.data(), bias.data(),
                        weights_packed.data());

#ifdef KAI_DEBUG
                    const size_t vec_length = kai_get_sme_vector_length_u32();
                    // Print packed weights - 1VL per row.
                    print_tensor(
                        {1, (weights_packed.size() / vec_length), 1, vec_length},
                        "\n Weights Packed :  ", weights_packed.data());
#endif
                    // -------------------------------------------------
                    // 3. Kernel takes in 6 rows of input and generates 4
                    //    rows of output across all channels at a time.
                    // -------------------------------------------------
                    constexpr size_t rows_handled = 4;  // no of rows kernel handles each time.
                    for (size_t out_row = 0; out_row < out_shape.h; out_row += rows_handled) {
                        // Variables below used to calculate start of input pointer.
                        const int start_in_row = out_row - padding.top;
                        const size_t pad_top = (start_in_row < 0) ? (-start_in_row) : 0;
                        const size_t in_row = (start_in_row < 0) ? 0 : start_in_row;

                        // Calculate row strides for pointer.
                        const size_t in_row_stride_bytes = (width * channels * sizeof(float));
                        const size_t out_row_stride_bytes = (out_shape.w * out_shape.c * sizeof(float));

                        // Number of input rows that can be read, number of output rows to calculate.
                        const size_t valid_input_rows = (in_row < height) ? (height - in_row) : 0;
                        const size_t valid_out_rows = (out_shape.h - out_row);

                        // Increment output/input pointers according to tile being calculated.
                        auto out_offset = kai_get_dst_offset_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(
                            out_row, out_row_stride_bytes);
                        auto in_offset = kai_get_src_offset_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(
                            in_row, in_row_stride_bytes);
                        const auto inptr = (uint8_t*)input.data() + in_offset;
                        auto outptr = (uint8_t*)out.data() + out_offset;

                        // NOTE: Kernel expects strides to be passed as bytes.
                        // f32_f32_f32p1vlx1b -> f32 output, f32 LHS, packed F32 rhs (with bias) as 1VL blocks.
                        // 3x3_s : 3x3 filter with stride 1
                        // 4xc : 4 rows across all output channels (plane c) is produced.
                        kai_run_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(
                            inptr, weights_packed.data(), outptr, in_row_stride_bytes, channels * sizeof(float),
                            out_row_stride_bytes, out_shape.c * sizeof(float), valid_input_rows, valid_out_rows,
                            padding.left, pad_top, 0.0f, clamp_min, clamp_max);
                    }

#ifdef KAI_DEBUG
                    // Print outputs
                    print_tensor(out_shape, "Reference : ", reinterpret_cast<float*>(ref.data()));
                    print_tensor(out_shape, "\n\n Actual : ", out.data());
                    std::cout << "\n\nOut shape : " << out_shape << std::endl;
#endif  // KAI_DEBUG

                    /// Check for mismatches in the tests.
                    size_t mismatches = 0;
                    for (size_t i = 0; i < out_shape.size(); i++) {
                        float ref_value = ref[i];
                        // FP32 rel tolerance - allows deviations of up to 0.05%
                        const auto err = (std::abs(out[i] - ref_value) / std::abs(ref_value));
                        if (err > 0.0005) {
                            std::cout << "Mismatches(Expected:Actual)" << ref_value << " : " << out[i] << std::endl;
                            mismatches++;
                        }
                        if (mismatches > 0) {
                            std::cout << "\nNumber of mismatches: " << mismatches << std::endl;
                        }
                    }
                }
            }
        }
    }
    std::cout << "total tests run: " << total_test << std::endl;
}
