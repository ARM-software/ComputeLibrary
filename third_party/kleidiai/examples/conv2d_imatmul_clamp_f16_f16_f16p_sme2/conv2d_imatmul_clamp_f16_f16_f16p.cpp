//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Example usage for Indirect GEMM with a convolution operation using two half-precision float matrices.
//

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || \
    !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_FP16.
#else  // Architectural features check.

#include <arm_fp16.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

// Include micro-kernel variants
#include "kai/ukernels/matmul/imatmul_clamp_f16_f16p_f16p/kai_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x16p2vlx2_x16p_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme.h"

using VEC_F16 = std::vector<float16_t>;

namespace {

constexpr float clamp_min = -9000.0F;
constexpr float clamp_max = 9000.0F;

struct Shape {
    size_t n;
    size_t h;
    size_t w;
    size_t c;
    [[nodiscard]] auto size() const -> size_t {
        return n * h * w * c;
    }

#ifdef KAI_DEBUG
    friend std::ostream& operator<<(std::ostream& os, const Shape& shape) {
        os << " [ " << shape.n << " , " << shape.h << " ," << shape.w << " , " << shape.c << " ] ";
        return os;
    }
#endif
};

/// Perform a convolution operation in nhwc data format.
/// @param[in] in_shape Shape of the input tensor in [N, H, W, C] DataFormat
/// @param[in] out_shape Shape of the output tensor in [N, H, W, C] DataFormat
/// @param[in] filter_height Height of convolution filter.
/// @param[in] filter_width Width of convolution filter.
/// @param[in] feature_map half float pointer to start of input tensor
/// @param[in] weights half float pointer to start of weights tensor
/// @param[in] bias half float pointer to start of bias tensor
/// @param[out] out half float pointer to start of output tensor
/// @param[in] clamp_min Minimum value to clamp final result
/// @param[in] clamp_max Max value to clamp final result
void convolution_layer_nhwc(
    const Shape& in_shape, const Shape& out_shape, const size_t filter_height, const size_t filter_width,
    const VEC_F16& feature_map, const VEC_F16& weights, const VEC_F16& bias, VEC_F16& out, float clamp_min,
    float clamp_max) {
    // We accumulate in FP32 and clamp later.
    std::vector<float> acc(out_shape.size());

    for (size_t n = 0; n < out_shape.n; ++n) {
        for (size_t out_h = 0; out_h < out_shape.h; ++out_h) {
            for (size_t out_w = 0; out_w < out_shape.w; ++out_w) {
                // Apply filter to feature map.
                for (size_t kernel_h = 0; kernel_h < filter_height; ++kernel_h) {
                    if (in_shape.h <= (out_h + kernel_h)) continue;
                    for (size_t kernel_w = 0; kernel_w < filter_width; ++kernel_w) {
                        if (in_shape.w <= (out_w + kernel_w)) continue;

                        for (size_t ic = 0; ic < in_shape.c; ++ic) {
                            auto in_idx =
                                ((n * in_shape.h + (out_h + kernel_h)) * in_shape.w + (out_w + kernel_w)) * in_shape.c +
                                ic;
                            auto weights_idx = (((kernel_h * filter_width + kernel_w) * in_shape.c + ic) * out_shape.c);
                            auto out_idx = ((n * out_shape.h + out_h) * out_shape.w + out_w) * out_shape.c;

                            for (size_t oc = 0; oc < out_shape.c; ++oc) {
                                // Perform actual accumulation and store in output vector
                                acc[out_idx + oc] += (feature_map[in_idx] * weights[weights_idx + oc]);
                            }
                        }
                    }
                }

                // Perform bias accumulation for channel idx and store in output vector.
                for (size_t oc = 0; oc < out_shape.c; ++oc) {
                    auto out_idx = ((n * out_shape.h + out_h) * out_shape.w + out_w) * out_shape.c;
                    acc[out_idx + oc] += bias[oc];
                }
            }
        }
    }

    // Apply clamping to accumulator, cast to FP16 and store in output vector at the same idx.
    for (size_t i = 0; i < out_shape.size(); i++) {
        out[i] = static_cast<float16_t>(std::clamp(acc[i], clamp_min, clamp_max));
    }
}

/// Fill a provided indirection table according to tensor shape parameters.
/// @param[in] feature_map Input feature map tensor
/// @param[out] indirection_table Indirection buffer to fill in place.
/// @param[in] pad_buffer Pointer to start of padding.
/// @param[in] in_shape Shape of input tensor [N,H,W,C] format.
/// @param[in] out_shape Shape of output tensor [N,H,W,C] format.
/// @param[in] filter_height Height of convolution filter.
/// @param[in] filter_width Width of convolution filter.
/// @param[in] itable_cols Number of columns in indirection table (m_step)
void init_indirection_table(
    const VEC_F16& feature_map, std::vector<const float16_t*>& indirect_table, const float16_t* pad_buffer,
    const Shape& in_shape, const Shape& out_shape, const size_t filter_height, const size_t filter_width,
    const size_t itable_cols) {
    // The indirection buffer here is a series of blocks each of size k_chunk_count * m_step.
    // Number of blocks is = round_up_division(M, m_step)
    const size_t block_size = filter_height * filter_width * itable_cols;
    const size_t in_hwc_size = in_shape.h * in_shape.w * in_shape.c;

    // The following code iterates over the first 3 dims of the output tensor and retrieves KH*KW number of pointers to
    // the input matrix for each idx. These pointers are stored columnwise in the itable, beginning with an offset.
    for (size_t batch_idx = 0; batch_idx < out_shape.n; batch_idx++) {
        for (size_t output_y = 0; output_y < out_shape.h; output_y++) {
            for (size_t output_x = 0; output_x < out_shape.w; output_x++) {
                // Calculates column and row offsets for itable index with respect to current block location and itable
                // column length (equivalent to m_step) The block start x/y offsets ensure the data is padded in the
                // format expected by the LHS Packing micro-kernel.
                size_t block_start_x =
                    (((batch_idx * out_shape.h * out_shape.w) + (output_y * out_shape.w + output_x)) % itable_cols);
                size_t block_start_y =
                    (((batch_idx * out_shape.h * out_shape.w) + (output_y * out_shape.w + output_x)) / itable_cols);
                for (size_t kernel_y = 0; kernel_y < filter_height; kernel_y++) {
                    const size_t input_y = output_y + kernel_y;
                    if (input_y < in_shape.h) {
                        for (size_t kernel_x = 0; kernel_x < filter_width; kernel_x++) {
                            size_t input_x = output_x + kernel_x;
                            size_t kernel_index = kernel_y * filter_width + kernel_x;
                            size_t index = (block_start_y * block_size) + block_start_x + kernel_index * itable_cols;

                            if (input_x < in_shape.w) {
                                indirect_table[index] =
                                    (feature_map.data() + batch_idx * in_hwc_size + input_y * in_shape.w * in_shape.c +
                                     input_x * in_shape.c);
                            } else {
                                indirect_table[index] = pad_buffer;
                            }
                        }
                    } else {
                        for (size_t kernel_x = 0; kernel_x < filter_width; kernel_x++) {
                            size_t kernel_index = kernel_y * filter_width + kernel_x;
                            size_t index = (block_start_y * block_size) + block_start_x + kernel_index * itable_cols;
                            indirect_table[index] = pad_buffer;
                        }
                    }
                }
            }
        }
    }
}

/// Fills the matrix with incremental values according to the provided weight.
/// @param[in] size Total number of elements to fill in passed vector;.
/// @param[in] dst Vector representing a tensor to fill.
/// @param[in] weight A weight value to increment by.
void fill_matrix(size_t size, VEC_F16& dst, const float16_t weight) {
    for (size_t i = 0; i < size; i++) {
        dst[i] = float16_t(i * weight);
    }
}

#ifdef KAI_DEBUG
/// Function prints a tensor in NHWC format.
/// Width and channels are printed on the same line. Square brackets are used to denote dimensions.
/// @param[in] shape A struct containing the NHWC shape of the tensor.
/// @param[in] name Name of the tensor
/// @param[in] src A vector of F16 elements representing the tensor.
void print_tensor(const Shape& shape, const char* name, const VEC_F16& src) {
    std::cout << name << " = [\n";
    for (size_t n = 0; n < shape.n; n++) {
        std::cout << "\n";
        for (size_t y = 0; y < shape.h; ++y) {
            std::cout << "  [";
            for (size_t x = 0; x < shape.w; x++) {
                std::cout << "[";
                for (size_t c = 0; c < shape.c; c++) {
                    if (c != 0) std::cout << " , ";
                    std::cout << std::setprecision(1) << std::fixed
                              << src[n * shape.h * shape.w * shape.c + y * shape.w * shape.c + x * shape.c + c];
                }
                std::cout << "] ";
            }
            std::cout << ("],\n");
        }
    }
    std::cout << ("]\n\n");
}
#endif  // KAI_DEBUG

// Verify the micro-kernel output matches the reference implementation
bool is_output_correct(
    size_t num_rows, size_t num_cols, const float16_t tolerance, const VEC_F16& ref, const VEC_F16& act) {
    bool is_valid = true;
    int count = 0;
    for (size_t i = 0; i < num_rows * num_cols; ++i) {
        if ((std::fabs((ref[i] - act[i]) / act[i])) > tolerance) {
            const size_t x = i % num_cols;
            const size_t y = i / num_cols;
            count++;
            std::cout << std::setprecision(5) << std::fixed << "ERROR![" << y << "][" << x << "]: ref=" << ref[i]
                      << " vs. act=" << act[i] << "\n";

            is_valid = false;
        }
    }
    std::cout << "\n\nThere are " << count << " mismatches." << std::endl;
    return is_valid;
}

size_t round_up_division(size_t a, size_t b) {
    return (a + b - 1) / b;
}
}  // namespace

int main() {
    // Arguments for convolution operation.
    // Padding must be valid
    const size_t batch_size = 5;
    const size_t input_height = 32;
    const size_t input_width = 32;
    const size_t input_channels = 3;
    const size_t filter_height = 5;
    const size_t filter_width = 2;
    const size_t out_channels = 2;

    // Use shape arguments to define tensor shapes in NHWC Format.
    const Shape in_shape{batch_size, input_height, input_width, input_channels};
    const Shape weights_shape{filter_height, filter_width, input_channels, out_channels};
    const Shape out_shape{
        batch_size, (input_height - filter_height + 1), (input_width - filter_width + 1), out_channels};

#ifdef KAI_DEBUG
    std::cout << "\nInput Shape : " << in_shape << " Kernel Shape : " << weights_shape
              << " Output Shape : " << out_shape << std::endl;
#endif  // KAI_DEBUG

    // Define and Fill Input Tensors for operation using shapes
    VEC_F16 feature_map(in_shape.size());
    VEC_F16 weights(weights_shape.size());
    VEC_F16 bias(out_channels);

    // Fill by iterating each element and incrementing each time by the provided weight, beginning at 0.
    fill_matrix(feature_map.size(), feature_map, 0.1f);
    fill_matrix(weights.size(), weights, 0.01f);
    fill_matrix(bias.size(), bias, 1.f);

    // The following are used as parameters in the indirection kernels
    const size_t out_nhw_size = out_shape.n * out_shape.h * out_shape.w;
    const size_t k_chunk_length = input_channels;
    const size_t k_chunk_count = filter_height * filter_width;

    // -------------------------------------------------
    // 1. Create Indirection buffer.
    // -------------------------------------------------
    // Define and Fill the indirection table in the format expected of the LHS Indirection Matmul kernel.
    // NOTE: out_nhw_size is equivalent to M argument for Indirection kernels.
    //       out_channels is equivalent to N argument for Indirection kernels.
    const size_t itable_cols = kai_get_m_step_lhs_imatmul_pack_x16p2vlx2_x16p_sme();
    const size_t itable_rows = k_chunk_count * round_up_division(out_nhw_size, itable_cols);
    std::vector<const float16_t*> indirect_table(itable_cols * itable_rows);

    // Padding buffer 'pad_buffer' is set to nullptr as there is no padding in this example.
    // Shapes specified are such that no padding should be needed.
    init_indirection_table(
        feature_map, indirect_table, nullptr, in_shape, out_shape, filter_height, filter_width, itable_cols);

    // -------------------------------------------------
    // 2. Pack LHS and RHS.
    // -------------------------------------------------
    auto lhs_packed_size_bytes =
        kai_get_lhs_packed_size_lhs_imatmul_pack_x16p2vlx2_x16p_sme(out_nhw_size, k_chunk_count, k_chunk_length);
    auto rhs_packed_size_bytes = kai_get_rhs_packed_size_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme(
        out_channels, k_chunk_count, k_chunk_length);

    VEC_F16 packed_lhs(lhs_packed_size_bytes / sizeof(float16_t));
    VEC_F16 packed_rhs(rhs_packed_size_bytes / sizeof(float16_t));

    // Padding is not used in the indirection buffer, therefore pad_ptr is nullptr
    // Ptr offset is provided as 0 as it is not needed to apply an offset to each valid pointer provided in the table in
    // this case.
    kai_run_lhs_imatmul_pack_x16p2vlx2_x16p_sme(
        out_nhw_size, k_chunk_count, k_chunk_length, (const void**)indirect_table.data(), 0, nullptr,
        packed_lhs.data());
    kai_run_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme(
        out_channels, k_chunk_count, k_chunk_length, out_channels * sizeof(float16_t), weights.data(), bias.data(),
        packed_rhs.data());

    // -------------------------------------------------
    // 3. Perform matmul operation and call reference, then compare.
    // -------------------------------------------------
    VEC_F16 act_output(out_shape.size());
    VEC_F16 ref_output(out_shape.size());

    kai_run_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa(
        out_nhw_size, out_channels, k_chunk_count, k_chunk_length, packed_lhs.data(), packed_rhs.data(),
        act_output.data(), out_channels * sizeof(float16_t), clamp_min, clamp_max);

    convolution_layer_nhwc(
        in_shape, out_shape, filter_height, filter_width, feature_map, weights, bias, ref_output, clamp_min, clamp_max);

#ifdef KAI_DEBUG
    print_tensor(out_shape, "\nTarget : ", act_output);
    print_tensor(out_shape, "\nREf : ", ref_output);
#endif  // KAI_DEBUG

    is_output_correct(out_nhw_size, out_channels, 0.0001f, ref_output, act_output);

    return 0;
}

#endif  // Architectural features check.
