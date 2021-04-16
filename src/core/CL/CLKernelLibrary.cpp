/*
 * Copyright (c) 2016-2021 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/core/CL/CLKernelLibrary.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Utils.h"
#include "support/StringSupport.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <utility>
#include <vector>

#ifdef ARM_COMPUTE_COMPRESSED_KERNELS
#include <zlib.h>

namespace
{
/* Decoding table */
constexpr std::array<uint8_t, 256> b64_invtab =
{
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 62, 0, 0, 0, 63,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 0, 0, 0, 0, 0,
    0, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};

/** Decode a base64 encoded string
 *
 * @param[in] str Base64 encoded string to decode
 *
 * @return The decode string in case of a valid, non-empty string otherwise an empty string
 */
std::string decode_base64(const std::string &str)
{
    constexpr const char pad_char = '=';

    // Handle empty string
    if(str.empty())
    {
        return {};
    }

    // Base64 encoded string has size multiple of 4
    if(str.length() % 4)
    {
        return {};
    }

    //
    // Check encoded string padding
    std::size_t padding = (str.rbegin()[0] == pad_char) + (str.rbegin()[1] == pad_char);
    const int   str_len = str.size();

    // Reserve memory for the decoded string
    // Note each 4 consecutive elements of 6-bit encode 3 bytes
    std::string dec_b64;
    dec_b64.reserve(((str_len / 4) * 3));

    // Block decoding function (exclude padding)
    int       c   = 0;
    const int end = str_len - 4 - padding;
    for(; c <= end; c += 4)
    {
        const int byte0 = b64_invtab[str[c]];
        const int byte1 = b64_invtab[str[c + 1]];
        const int byte2 = b64_invtab[str[c + 2]];
        const int byte3 = b64_invtab[str[c + 3]];

        dec_b64.push_back((byte0 << 2) | (byte1 >> 4));
        dec_b64.push_back((byte1 << 4) | (byte2 >> 2));
        dec_b64.push_back((byte2 << 6) | (byte3));
    }

    // Last step that might contain padding symbols
    if(padding == 1)
    {
        const int byte0 = b64_invtab[str[c]];
        const int byte1 = b64_invtab[str[c + 1]];
        const int byte2 = b64_invtab[str[c + 2]];

        dec_b64.push_back((byte0 << 2) | (byte1 >> 4));
        dec_b64.push_back((byte1 << 4) | (byte2 >> 2));
    }
    else if(padding == 2)
    {
        const int byte0 = b64_invtab[str[c]];
        const int byte1 = b64_invtab[str[c + 1]];

        dec_b64.push_back((byte0 << 2) | (byte1 >> 4));
    }

    return dec_b64;
}

/** Decompress a zlib compressed string
 *
 * @param[in] str ZLib compressed string
 *
 * @return The decompressed string if successful, otherwise false.
 */
std::string decompress_zlib(const std::string &str)
{
    // Create and initialize decompression stream
    z_stream ds{};
    if(inflateInit(&ds) != Z_OK)
    {
        return std::string();
    }
    ds.avail_in = str.size();
    ds.next_in  = (Bytef *)str.data();

    // Roll-over the string using a buffer and decompress
    int         status = Z_OK;
    char        roll_buff[16384];
    std::string inflated_str;
    do
    {
        ds.avail_out = sizeof(roll_buff);
        ds.next_out  = reinterpret_cast<Bytef *>(roll_buff);

        status = inflate(&ds, 0);
        if(inflated_str.size() < ds.total_out)
        {
            inflated_str.append(roll_buff, ds.total_out - inflated_str.size());
        }
    }
    while(status == Z_OK);

    // Finalize decompression stream
    inflateEnd(&ds);
    if(status != Z_STREAM_END)
    {
        return std::string();
    }

    return inflated_str;
}
} // namespace
#endif /* ARM_COMPUTE_COMPRESSED_KERNELS */

using namespace arm_compute;
const std::map<std::string, std::string> CLKernelLibrary::_kernel_program_map =
{
    { "activation_layer", "activation_layer.cl" },
    { "activation_layer_quant", "activation_layer_quant.cl" },
    { "activation_layer_quant_f32", "activation_layer_quant.cl" },
    { "arg_min_max_x", "arg_min_max.cl" },
    { "arg_min_max_y", "arg_min_max.cl" },
    { "arg_min_max_z", "arg_min_max.cl" },
    { "arg_min_max_w", "arg_min_max.cl" },
    { "batch_to_space_nchw", "batch_to_space.cl" },
    { "batch_to_space_static_nchw", "batch_to_space.cl" },
    { "batch_to_space_nhwc", "batch_to_space.cl" },
    { "batch_to_space_static_nhwc", "batch_to_space.cl" },
    { "batchnormalization_layer_nchw", "batchnormalization_layer.cl" },
    { "batchnormalization_layer_nhwc", "batchnormalization_layer.cl" },
    { "bitwise_or", "bitwise_op.cl" },
    { "bitwise_and", "bitwise_op.cl" },
    { "bitwise_xor", "bitwise_op.cl" },
    { "bitwise_not", "bitwise_op.cl" },
    { "bounding_box_transform", "bounding_box_transform.cl" },
    { "bounding_box_transform_quantized", "bounding_box_transform_quantized.cl" },
    { "channel_shuffle_nchw", "channel_shuffle.cl" },
    { "channel_shuffle_nhwc", "channel_shuffle.cl" },
    { "compare_equal", "comparisons.cl" },
    { "compare_equal_quantized", "comparisons.cl" },
    { "compare_notequal", "comparisons.cl" },
    { "compare_notequal_quantized", "comparisons.cl" },
    { "compare_greater", "comparisons.cl" },
    { "compare_greater_quantized", "comparisons.cl" },
    { "compare_greaterequal", "comparisons.cl" },
    { "compare_greaterequal_quantized", "comparisons.cl" },
    { "compare_less", "comparisons.cl" },
    { "compare_less_quantized", "comparisons.cl" },
    { "compare_lessequal", "comparisons.cl" },
    { "compare_lessequal_quantized", "comparisons.cl" },
    { "concatenate", "concatenate.cl" },
    { "concatenate_width", "concatenate.cl" },
    { "concatenate_height", "concatenate.cl" },
    { "concatenate_width_x2", "concatenate.cl" },
    { "concatenate_width_x4", "concatenate.cl" },
    { "col2im", "col2im.cl" },
    { "convert_depth_down", "depth_convert.cl" },
    { "convert_depth_up", "depth_convert.cl" },
    { "convert_fc_weights", "convert_fc_weights.cl" },
    { "copy_tensor", "copy_tensor.cl" },
    { "crop_tensor", "crop_tensor.cl" },
    { "deconvolution_reshape", "deconvolution_layer.cl" },
    { "deconvolution_upsample", "deconvolution_layer.cl" },
    { "depthwise_convolution_3x3", "depthwise_convolution.cl" },
    { "depthwise_convolution_3x3_f16", "depthwise_convolution.cl" },
    { "depthwise_convolution_3x3_nhwc", "depthwise_convolution.cl" },
    { "depthwise_convolution_3x3_nhwc_stride1", "depthwise_convolution.cl" },
    { "dwc_MxN_native_fp_nhwc", "depthwise_convolution.cl" },
    { "dwc_MxN_native_quantized8_nhwc", "depthwise_convolution_quantized.cl" },
    { "dwc_3x3_native_quantized8_nchw", "depthwise_convolution_quantized.cl" },
    { "dwc_3x3_native_quantized8_dot8_nchw", "depthwise_convolution_quantized.cl" },
    { "depth_to_space_nchw", "depth_to_space.cl" },
    { "depth_to_space_nhwc", "depth_to_space.cl" },
    { "depthwise_convolution_3x3_stridex1_stridey1_f16", "depthwise_convolution.cl" },
    { "depthwise_convolution_3x3_stridex2_stridey2_f16", "depthwise_convolution.cl" },
    { "depthwise_convolution_3x3_stridex1_stridey1_f32", "depthwise_convolution.cl" },
    { "depthwise_convolution_3x3_stridex2_stridey2_f32", "depthwise_convolution.cl" },
    { "dequantization_layer", "dequantization_layer.cl" },
    { "dequantization_layer_per_channel_nhwc", "dequantization_layer.cl" },
    { "dequantization_layer_per_channel_nchw", "dequantization_layer.cl" },
    { "direct_convolution_nhwc", "direct_convolution.cl" },
    { "direct_convolution1x1", "direct_convolution1x1.cl" },
    { "direct_convolution1x1_f32_bifrost", "direct_convolution1x1.cl" },
    { "direct_convolution3x3", "direct_convolution3x3.cl" },
    { "direct_convolution3x3_f32_bifrost", "direct_convolution3x3.cl" },
    { "direct_convolution5x5", "direct_convolution5x5.cl" },
    { "direct_convolution5x5_f32_bifrost", "direct_convolution5x5.cl" },
    { "direct_convolution_quantized", "direct_convolution_quantized.cl" },
    { "elementwise_operation_ADD", "elementwise_operation.cl" },
    { "elementwise_operation_SUB", "elementwise_operation.cl" },
    { "elementwise_operation_MAX", "elementwise_operation.cl" },
    { "elementwise_operation_MIN", "elementwise_operation.cl" },
    { "elementwise_operation_DIV", "elementwise_operation.cl" },
    { "elementwise_operation_SQUARED_DIFF", "elementwise_operation.cl" },
    { "elementwise_operation_POWER", "elementwise_operation.cl" },
    { "elementwise_operation_PRELU", "elementwise_operation.cl" },
    { "elementwise_operation_AND", "elementwise_operation.cl" },
    { "elementwise_operation_OR", "elementwise_operation.cl" },
    { "elementwise_operation_ADD_quantized", "elementwise_operation_quantized.cl" },
    { "elementwise_operation_SUB_quantized", "elementwise_operation_quantized.cl" },
    { "elementwise_operation_MAX_quantized", "elementwise_operation_quantized.cl" },
    { "elementwise_operation_MIN_quantized", "elementwise_operation_quantized.cl" },
    { "elementwise_operation_DIV_quantized", "elementwise_operation_quantized.cl" },
    { "elementwise_operation_SQUARED_DIFF_quantized", "elementwise_operation_quantized.cl" },
    { "elementwise_operation_PRELU_quantized", "elementwise_operation_quantized.cl" },
    { "elementwise_unary", "elementwise_unary.cl" },
    { "fft_digit_reverse_axis_0", "fft_digit_reverse.cl" },
    { "fft_digit_reverse_axis_1", "fft_digit_reverse.cl" },
    { "fft_radix_2_first_stage_axis_0", "fft.cl" },
    { "fft_radix_2_first_stage_axis_1", "fft.cl" },
    { "fft_radix_2_axis_0", "fft.cl" },
    { "fft_radix_2_axis_1", "fft.cl" },
    { "fft_radix_3_first_stage_axis_0", "fft.cl" },
    { "fft_radix_3_first_stage_axis_1", "fft.cl" },
    { "fft_radix_3_axis_0", "fft.cl" },
    { "fft_radix_3_axis_1", "fft.cl" },
    { "fft_radix_4_first_stage_axis_0", "fft.cl" },
    { "fft_radix_4_first_stage_axis_1", "fft.cl" },
    { "fft_radix_4_axis_0", "fft.cl" },
    { "fft_radix_4_axis_1", "fft.cl" },
    { "fft_radix_5_first_stage_axis_0", "fft.cl" },
    { "fft_radix_5_first_stage_axis_1", "fft.cl" },
    { "fft_radix_5_axis_0", "fft.cl" },
    { "fft_radix_5_axis_1", "fft.cl" },
    { "fft_radix_7_first_stage_axis_0", "fft.cl" },
    { "fft_radix_7_first_stage_axis_1", "fft.cl" },
    { "fft_radix_7_axis_0", "fft.cl" },
    { "fft_radix_7_axis_1", "fft.cl" },
    { "fft_radix_8_first_stage_axis_0", "fft.cl" },
    { "fft_radix_8_first_stage_axis_1", "fft.cl" },
    { "fft_radix_8_axis_0", "fft.cl" },
    { "fft_radix_8_axis_1", "fft.cl" },
    { "fft_scale_conj", "fft_scale.cl" },
    { "fill_image_borders_constant", "fill_border.cl" },
    { "fill_image_borders_replicate", "fill_border.cl" },
    { "floor_layer", "floor.cl" },
    { "fuse_batchnormalization_layer", "batchnormalization_layer.cl" },
    { "gather", "gather.cl" },
    { "gemm_ma_f16", "gemm.cl" },
    { "gemm_ma_f32", "gemm.cl" },
    { "gemm_mv", "gemv.cl" },
    { "gemm_mv_quantized", "gemv.cl" },
    { "gemm_mm_interleaved_transposed_f16", "gemm_v1.cl" },
    { "gemm_mm_interleaved_transposed_f16_acc32", "gemm_v1.cl" },
    { "gemm_mm_interleaved_transposed_f16_bifrost", "gemm_v1.cl" },
    { "gemm_mm_interleaved_transposed_f32", "gemm_v1.cl" },
    { "gemm_mm_interleaved_transposed_f32_bifrost", "gemm_v1.cl" },
    { "gemm_mm_floating_point", "gemm_v1.cl" },
    { "gemm_mm_floating_point_f16_bifrost", "gemm_v1.cl" },
    { "gemm_mm_floating_point_f16_bifrost_acc32", "gemm_v1.cl" },
    { "gemm_mm_floating_point_f32_bifrost", "gemm_v1.cl" },
    { "gemm_mm_floating_point_f32_bifrost_1000", "gemm_v1.cl" },
    { "gemm_mm_native", "gemm.cl" },
    { "gemm_mm_reshaped_lhs_nt_rhs_t", "gemm.cl" },
    { "gemm_mm_reshaped_lhs_nt_rhs_t_texture", "gemm.cl" },
    { "gemm_mm_reshaped_lhs_t_rhs_nt", "gemm.cl" },
    { "gemm_mm_reshaped_lhs_t_rhs_nt_texture", "gemm.cl" },
    { "gemm_mm_reshaped_only_rhs_nt", "gemm.cl" },
    { "gemm_mm_reshaped_only_rhs_nt_texture", "gemm.cl" },
    { "gemm_mm_reshaped_only_rhs_t", "gemm.cl" },
    { "gemm_mm_reshaped_only_rhs_t_texture", "gemm.cl" },
    { "gemm_lc_vm_f32", "gemm.cl" },
    { "gemm_reshape_lhs_matrix_nt", "gemm.cl" },
    { "gemm_reshape_lhs_matrix_t", "gemm.cl" },
    { "gemm_reshape_rhs_matrix_nt", "gemm.cl" },
    { "gemm_reshape_rhs_matrix_t", "gemm.cl" },
    { "gemmlowp_matrix_a_reduction", "gemmlowp.cl" },
    { "gemmlowp_matrix_a_reduction_dot8", "gemmlowp.cl" },
    { "gemmlowp_matrix_b_reduction", "gemmlowp.cl" },
    { "gemmlowp_mm_native", "gemmlowp.cl" },
    { "gemmlowp_mm_reshaped_lhs_nt_rhs_t", "gemmlowp.cl" },
    { "gemmlowp_mm_reshaped_only_rhs_t", "gemmlowp.cl" },
    { "gemmlowp_mm_reshaped_only_rhs_t_fused_output_stage_fixedpoint", "gemmlowp.cl" },
    { "gemmlowp_offset_contribution", "gemmlowp.cl" },
    { "gemmlowp_offset_contribution_quantize_down", "gemmlowp.cl" },
    { "gemmlowp_offset_contribution_quantize_down_fixedpoint", "gemmlowp.cl" },
    { "gemmlowp_output_stage_quantize_down", "gemmlowp.cl" },
    { "gemmlowp_output_stage_quantize_down_fixedpoint", "gemmlowp.cl" },
    { "gemmlowp_output_stage_quantize_down_fixedpoint_qsymm16", "gemmlowp.cl" },
    { "gemmlowp_output_stage_quantize_down_float", "gemmlowp.cl" },
    { "generate_proposals_compute_all_anchors", "generate_proposals.cl" },
    { "generate_proposals_compute_all_anchors_quantized", "generate_proposals_quantized.cl" },
    { "im2col1x1_stridex1_nchw", "im2col.cl" },
    { "im2col3x3_nchw", "im2col.cl" },
    { "im2col5x5_nchw", "im2col.cl" },
    { "im2col11x11_padx0_pady0_nchw", "im2col.cl" },
    { "im2col_generic_nchw", "im2col.cl" },
    { "im2col_generic_padx0_pady0_nchw", "im2col.cl" },
    { "im2col3x3_nhwc", "im2col.cl" },
    { "im2col9x9_nhwc", "im2col.cl" },
    { "im2col_generic_nhwc", "im2col.cl" },
    { "instance_normalization", "instance_normalization.cl" },
    { "compute_mean_var", "instance_normalization.cl" },
    { "l2_normalize_x", "l2_normalize.cl" },
    { "l2_normalize_y", "l2_normalize.cl" },
    { "l2_normalize_z", "l2_normalize.cl" },
    { "max_unpooling_layer_2", "unpooling_layer.cl" },
    { "mean_stddev_normalization", "mean_stddev_normalization.cl" },
    { "memset", "memset.cl" },
    { "minmax_layer", "minmax_layer.cl" },
    { "non_max_suppression", "nonmax.cl" },
    { "normalization_layer_cross_map_nchw", "normalization_layer.cl" },
    { "normalization_layer_cross_map_nhwc", "normalization_layer.cl" },
    { "normalization_layer_in_map_nchw", "normalization_layer.cl" },
    { "normalization_layer_in_map_nhwc", "normalization_layer.cl" },
    { "normalize_planar_yuv_layer_nchw", "normalize_planar_yuv_layer.cl" },
    { "normalize_planar_yuv_layer_nhwc", "normalize_planar_yuv_layer.cl" },
    { "normalize_planar_yuv_layer_q8_nchw", "normalize_planar_yuv_layer_quantized.cl" },
    { "normalize_planar_yuv_layer_q8_nhwc", "normalize_planar_yuv_layer_quantized.cl" },
    { "pad_layer_constant", "pad_layer.cl" },
    { "pad_layer_symmetric_reflect", "pad_layer.cl" },
    { "permute", "permute.cl" },
    { "pixelwise_mul_complex", "pixelwise_mul_float.cl" },
    { "pixelwise_mul_float", "pixelwise_mul_float.cl" },
    { "pixelwise_mul_int", "pixelwise_mul_int.cl" },
    { "pixelwise_mul_quantized", "pixelwise_mul_int.cl" },
    { "pooling_layer_2", "pooling_layer.cl" },
    { "pooling_layer_3", "pooling_layer.cl" },
    { "pooling_layer_optimized_3", "pooling_layer.cl" },
    { "pooling_layer_7", "pooling_layer.cl" },
    { "pooling_layer_MxN_nchw", "pooling_layer.cl" },
    { "pooling_layer_MxN_nhwc", "pooling_layer.cl" },
    { "pooling_layer_2x2_nhwc", "pooling_layer.cl" },
    { "pooling_layer_2_nchw_indices_fp32", "pooling_layer.cl" },
    { "pooling_layer_2_nchw_indices_fp16", "pooling_layer.cl" },
    { "pooling_layer_MxN_quantized_nhwc", "pooling_layer_quantized.cl" },
    { "pooling_layer_MxN_quantized_nchw", "pooling_layer_quantized.cl" },
    { "prior_box_layer_nchw", "prior_box_layer.cl" },
    { "qlstm_layer_normalization", "qlstm_layer_normalization.cl" },
    { "quantization_layer", "quantization_layer.cl" },
    { "range", "range.cl" },
    { "range_quantized", "range.cl" },
    { "reduction_operation_x", "reduction_operation.cl" },
    { "reduction_operation_non_parallel_x", "reduction_operation.cl" },
    { "reduction_operation_y", "reduction_operation.cl" },
    { "reduction_operation_z", "reduction_operation.cl" },
    { "reduction_operation_w", "reduction_operation.cl" },
    { "remap_nearest_neighbour", "remap.cl" },
    { "remap_bilinear", "remap.cl" },
    { "reorg_layer_nchw", "reorg_layer.cl" },
    { "reorg_layer_nhwc", "reorg_layer.cl" },
    { "reshape_layer", "reshape_layer.cl" },
    { "reshape_to_columns", "convolution_layer.cl" },
    { "reverse", "reverse.cl" },
    { "roi_align_layer", "roi_align_layer.cl" },
    { "roi_align_layer_quantized", "roi_align_layer_quantized.cl" },
    { "roi_pooling_layer", "roi_pooling_layer.cl" },
    { "scale_nearest_neighbour_nchw", "scale.cl" },
    { "scale_nearest_neighbour_nhwc", "scale.cl" },
    { "scale_bilinear_nchw", "scale.cl" },
    { "scale_bilinear_nhwc", "scale.cl" },
    { "scale_bilinear_quantized_nchw", "scale_quantized.cl" },
    { "scale_bilinear_quantized_nhwc", "scale_quantized.cl" },
    { "select_same_rank", "select.cl" },
    { "select_different_rank_2", "select.cl" },
    { "select_different_rank_n", "select.cl" },
    { "softmax_layer_norm", "softmax_layer.cl" },
    { "softmax_layer_norm_quantized", "softmax_layer_quantized.cl" },
    { "softmax_layer_max_shift_exp_sum_quantized_serial", "softmax_layer_quantized.cl" },
    { "softmax_layer_max_shift_exp_sum_quantized_parallel", "softmax_layer_quantized.cl" },
    { "softmax_layer_max_shift_exp_sum_serial", "softmax_layer.cl" },
    { "space_to_batch_nchw", "space_to_batch.cl" },
    { "space_to_batch_static_nchw", "space_to_batch.cl" },
    { "space_to_batch_nhwc", "space_to_batch.cl" },
    { "space_to_batch_static_nhwc", "space_to_batch.cl" },
    { "space_to_depth_nchw", "space_to_depth.cl" },
    { "space_to_depth_nhwc", "space_to_depth.cl" },
    { "softmax_layer_max_shift_exp_sum_parallel", "softmax_layer.cl" },
    { "stack_layer", "stack_layer.cl" },
    { "strided_slice", "slice_ops.cl" },
    { "tile", "tile.cl" },
    { "transpose", "transpose.cl" },
    { "upsample_layer_nchw", "upsample_layer.cl" },
    { "upsample_layer_nhwc", "upsample_layer.cl" },
    { "winograd_filter_transform_2x2_3x3_nchw", "winograd_filter_transform.cl" },
    { "winograd_filter_transform_2x1_3x1_nchw", "winograd_filter_transform.cl" },
    { "winograd_filter_transform_1x2_1x3_nchw", "winograd_filter_transform.cl" },
    { "winograd_filter_transform_4x4_3x3_nchw", "winograd_filter_transform.cl" },
    { "winograd_filter_transform_4x1_3x1_nchw", "winograd_filter_transform.cl" },
    { "winograd_filter_transform_1x4_1x3_nchw", "winograd_filter_transform.cl" },
    { "winograd_filter_transform_4x4_5x5_nchw", "winograd_filter_transform.cl" },
    { "winograd_filter_transform_4x1_5x1_nchw", "winograd_filter_transform.cl" },
    { "winograd_filter_transform_1x4_1x5_nchw", "winograd_filter_transform.cl" },
    { "winograd_filter_transform_4x1_3x1_nhwc", "winograd_filter_transform.cl" },
    { "winograd_filter_transform_1x4_1x3_nhwc", "winograd_filter_transform.cl" },
    { "winograd_filter_transform_4x4_3x3_nhwc", "winograd_filter_transform.cl" },
    { "winograd_filter_transform_4x4_5x5_nhwc", "winograd_filter_transform.cl" },
    { "winograd_filter_transform_4x1_5x1_nhwc", "winograd_filter_transform.cl" },
    { "winograd_filter_transform_1x4_1x5_nhwc", "winograd_filter_transform.cl" },
    { "winograd_filter_transform_2x2_7x7_nhwc", "winograd_filter_transform.cl" },
    { "winograd_filter_transform_2x1_7x1_nhwc", "winograd_filter_transform.cl" },
    { "winograd_filter_transform_1x2_1x7_nhwc", "winograd_filter_transform.cl" },
    { "winograd_input_transform_2x2_3x3_stepz1_nchw", "winograd_input_transform.cl" },
    { "winograd_input_transform_2x2_3x3_stepz2_nchw", "winograd_input_transform.cl" },
    { "winograd_input_transform_2x1_3x1_stepz1_nchw", "winograd_input_transform.cl" },
    { "winograd_input_transform_2x1_3x1_stepz2_nchw", "winograd_input_transform.cl" },
    { "winograd_input_transform_1x2_1x3_stepz1_nchw", "winograd_input_transform.cl" },
    { "winograd_input_transform_1x2_1x3_stepz2_nchw", "winograd_input_transform.cl" },
    { "winograd_input_transform_4x4_3x3_stepz1_nchw", "winograd_input_transform.cl" },
    { "winograd_input_transform_4x1_3x1_stepz1_nchw", "winograd_input_transform.cl" },
    { "winograd_input_transform_1x4_1x3_stepz1_nchw", "winograd_input_transform.cl" },
    { "winograd_input_transform_4x4_5x5_stepz1_nchw", "winograd_input_transform.cl" },
    { "winograd_input_transform_4x1_5x1_stepz1_nchw", "winograd_input_transform.cl" },
    { "winograd_input_transform_1x4_1x5_stepz1_nchw", "winograd_input_transform.cl" },
    { "winograd_input_transform_4x1_3x1_stepz1_nhwc", "winograd_input_transform.cl" },
    { "winograd_input_transform_1x4_1x3_stepz1_nhwc", "winograd_input_transform.cl" },
    { "winograd_input_transform_4x4_3x3_stepz1_nhwc", "winograd_input_transform.cl" },
    { "winograd_input_transform_4x4_5x5_stepz1_nhwc", "winograd_input_transform.cl" },
    { "winograd_input_transform_4x1_5x1_stepz1_nhwc", "winograd_input_transform.cl" },
    { "winograd_input_transform_1x4_1x5_stepz1_nhwc", "winograd_input_transform.cl" },
    { "winograd_input_transform_2x2_7x7_stepz1_nhwc", "winograd_input_transform.cl" },
    { "winograd_input_transform_2x1_7x1_stepz1_nhwc", "winograd_input_transform.cl" },
    { "winograd_input_transform_1x2_1x7_stepz1_nhwc", "winograd_input_transform.cl" },
    { "winograd_output_transform_2x2_3x3_nchw", "winograd_output_transform.cl" },
    { "winograd_output_transform_2x1_3x1_nchw", "winograd_output_transform.cl" },
    { "winograd_output_transform_1x2_1x3_nchw", "winograd_output_transform.cl" },
    { "winograd_output_transform_4x4_3x3_nchw", "winograd_output_transform.cl" },
    { "winograd_output_transform_4x1_3x1_nchw", "winograd_output_transform.cl" },
    { "winograd_output_transform_1x4_1x3_nchw", "winograd_output_transform.cl" },
    { "winograd_output_transform_4x4_5x5_nchw", "winograd_output_transform.cl" },
    { "winograd_output_transform_4x1_5x1_nchw", "winograd_output_transform.cl" },
    { "winograd_output_transform_1x4_1x5_nchw", "winograd_output_transform.cl" },
    { "winograd_output_transform_4x1_3x1_nhwc", "winograd_output_transform.cl" },
    { "winograd_output_transform_1x4_1x3_nhwc", "winograd_output_transform.cl" },
    { "winograd_output_transform_4x4_3x3_nhwc", "winograd_output_transform.cl" },
    { "winograd_output_transform_4x4_5x5_nhwc", "winograd_output_transform.cl" },
    { "winograd_output_transform_4x1_5x1_nhwc", "winograd_output_transform.cl" },
    { "winograd_output_transform_1x4_1x5_nhwc", "winograd_output_transform.cl" },
    { "winograd_output_transform_2x2_7x7_nhwc", "winograd_output_transform.cl" },
    { "winograd_output_transform_2x1_7x1_nhwc", "winograd_output_transform.cl" },
    { "winograd_output_transform_1x2_1x7_nhwc", "winograd_output_transform.cl" },
    { "yolo_layer_nchw", "yolo_layer.cl" },
    { "yolo_layer_nhwc", "yolo_layer.cl" },
};

const std::map<std::string, std::string> CLKernelLibrary::_program_source_map =
{
#ifdef EMBEDDED_KERNELS
    {
        "activation_layer.cl",
#include "./cl_kernels/activation_layer.clembed"
    },
    {
        "activation_layer_quant.cl",
#include "./cl_kernels/activation_layer_quant.clembed"
    },
    {
        "arg_min_max.cl",
#include "./cl_kernels/arg_min_max.clembed"
    },
    {
        "batch_to_space.cl",
#include "./cl_kernels/batch_to_space.clembed"
    },
    {
        "bitwise_op.cl",
#include "./cl_kernels/bitwise_op.clembed"
    },
    {
        "bounding_box_transform.cl",
#include "./cl_kernels/bounding_box_transform.clembed"
    },
    {
        "bounding_box_transform_quantized.cl",
#include "./cl_kernels/bounding_box_transform_quantized.clembed"
    },
    {
        "channel_shuffle.cl",
#include "./cl_kernels/channel_shuffle.clembed"
    },
    {
        "col2im.cl",
#include "./cl_kernels/col2im.clembed"
    },
    {
        "comparisons.cl",
#include "./cl_kernels/comparisons.clembed"
    },
    {
        "concatenate.cl",
#include "./cl_kernels/concatenate.clembed"
    },
    {
        "convert_fc_weights.cl",
#include "./cl_kernels/convert_fc_weights.clembed"
    },
    {
        "convolution_layer.cl",
#include "./cl_kernels/convolution_layer.clembed"
    },
    {
        "copy_tensor.cl",
#include "./cl_kernels/copy_tensor.clembed"
    },
    {
        "crop_tensor.cl",
#include "./cl_kernels/crop_tensor.clembed"
    },
    {
        "upsample_layer.cl",
#include "./cl_kernels/upsample_layer.clembed"
    },
    {
        "deconvolution_layer.cl",
#include "./cl_kernels/deconvolution_layer.clembed"
    },
    {
        "depth_convert.cl",
#include "./cl_kernels/depth_convert.clembed"
    },
    {
        "depth_to_space.cl",
#include "./cl_kernels/depth_to_space.clembed"
    },
    {
        "depthwise_convolution.cl",
#include "./cl_kernels/depthwise_convolution.clembed"
    },
    {
        "depthwise_convolution_quantized.cl",
#include "./cl_kernels/depthwise_convolution_quantized.clembed"
    },
    {
        "dequantization_layer.cl",
#include "./cl_kernels/dequantization_layer.clembed"
    },
    {
        "direct_convolution1x1.cl",
#include "./cl_kernels/direct_convolution1x1.clembed"
    },
    {
        "direct_convolution3x3.cl",
#include "./cl_kernels/direct_convolution3x3.clembed"
    },
    {
        "direct_convolution5x5.cl",
#include "./cl_kernels/direct_convolution5x5.clembed"
    },
    {
        "direct_convolution_quantized.cl",
#include "./cl_kernels/direct_convolution_quantized.clembed"
    },
    {
        "direct_convolution.cl",
#include "./cl_kernels/direct_convolution.clembed"
    },
    {
        "elementwise_operation.cl",
#include "./cl_kernels/elementwise_operation.clembed"
    },
    {
        "elementwise_operation_quantized.cl",
#include "./cl_kernels/elementwise_operation_quantized.clembed"
    },
    {
        "elementwise_unary.cl",
#include "./cl_kernels/elementwise_unary.clembed"
    },
    {
        "fft.cl",
#include "./cl_kernels/fft.clembed"
    },
    {
        "fft_digit_reverse.cl",
#include "./cl_kernels/fft_digit_reverse.clembed"
    },
    {
        "fft_scale.cl",
#include "./cl_kernels/fft_scale.clembed"
    },
    {
        "fill_border.cl",
#include "./cl_kernels/fill_border.clembed"
    },
    {
        "floor.cl",
#include "./cl_kernels/floor.clembed"
    },
    {
        "gather.cl",
#include "./cl_kernels/gather.clembed"
    },
    {
        "gemm.cl",
#include "./cl_kernels/gemm.clembed"
    },
    {
        "gemm_v1.cl",
#include "./cl_kernels/gemm_v1.clembed"
    },
    {
        "gemmlowp.cl",
#include "./cl_kernels/gemmlowp.clembed"
    },
    {
        "gemv.cl",
#include "./cl_kernels/gemv.clembed"
    },
    {
        "generate_proposals.cl",
#include "./cl_kernels/generate_proposals.clembed"
    },
    {
        "generate_proposals_quantized.cl",
#include "./cl_kernels/generate_proposals_quantized.clembed"
    },
    {
        "helpers.h",
#include "./cl_kernels/helpers.hembed"
    },
    {
        "helpers_asymm.h",
#include "./cl_kernels/helpers_asymm.hembed"
    },
    {
        "im2col.cl",
#include "./cl_kernels/im2col.clembed"
    },
    {
        "instance_normalization.cl",
#include "./cl_kernels/instance_normalization.clembed"
    },
    {
        "l2_normalize.cl",
#include "./cl_kernels/l2_normalize.clembed"
    },
    {
        "mean_stddev_normalization.cl",
#include "./cl_kernels/mean_stddev_normalization.clembed"
    },
    {
        "memset.cl",
#include "./cl_kernels/memset.clembed"
    },
    {
        "minmax_layer.cl",
#include "./cl_kernels/minmax_layer.clembed"
    },
    {
        "nonmax.cl",
#include "./cl_kernels/nonmax.clembed"
    },
    {
        "normalization_layer.cl",
#include "./cl_kernels/normalization_layer.clembed"
    },
    {
        "normalize_planar_yuv_layer.cl",
#include "./cl_kernels/normalize_planar_yuv_layer.clembed"
    },
    {
        "normalize_planar_yuv_layer_quantized.cl",
#include "./cl_kernels/normalize_planar_yuv_layer_quantized.clembed"
    },
    {
        "batchnormalization_layer.cl",
#include "./cl_kernels/batchnormalization_layer.clembed"
    },
    {
        "pad_layer.cl",
#include "./cl_kernels/pad_layer.clembed"
    },
    {
        "permute.cl",
#include "./cl_kernels/permute.clembed"
    },
    {
        "pixelwise_mul_float.cl",
#include "./cl_kernels/pixelwise_mul_float.clembed"
    },
    {
        "pixelwise_mul_int.cl",
#include "./cl_kernels/pixelwise_mul_int.clembed"
    },
    {
        "pooling_layer.cl",
#include "./cl_kernels/pooling_layer.clembed"
    },
    {
        "pooling_layer_quantized.cl",
#include "./cl_kernels/pooling_layer_quantized.clembed"
    },
    {
        "prior_box_layer.cl",
#include "./cl_kernels/prior_box_layer.clembed"
    },
    {
        "qlstm_layer_normalization.cl",
#include "./cl_kernels/qlstm_layer_normalization.clembed"
    },
    {
        "quantization_layer.cl",
#include "./cl_kernels/quantization_layer.clembed"
    },
    {
        "range.cl",
#include "./cl_kernels/range.clembed"
    },
    {
        "reduction_operation.cl",
#include "./cl_kernels/reduction_operation.clembed"
    },
    {
        "remap.cl",
#include "./cl_kernels/remap.clembed"
    },
    {
        "reorg_layer.cl",
#include "./cl_kernels/reorg_layer.clembed"
    },
    {
        "reshape_layer.cl",
#include "./cl_kernels/reshape_layer.clembed"
    },
    {
        "reverse.cl",
#include "./cl_kernels/reverse.clembed"
    },
    {
        "roi_align_layer.cl",
#include "./cl_kernels/roi_align_layer.clembed"
    },
    {
        "roi_align_layer_quantized.cl",
#include "./cl_kernels/roi_align_layer_quantized.clembed"
    },
    {
        "roi_pooling_layer.cl",
#include "./cl_kernels/roi_pooling_layer.clembed"
    },
    {
        "scale.cl",
#include "./cl_kernels/scale.clembed"
    },
    {
        "scale_quantized.cl",
#include "./cl_kernels/scale_quantized.clembed"
    },
    {
        "select.cl",
#include "./cl_kernels/select.clembed"
    },
    {
        "softmax_layer.cl",
#include "./cl_kernels/softmax_layer.clembed"
    },
    {
        "softmax_layer_quantized.cl",
#include "./cl_kernels/softmax_layer_quantized.clembed"
    },
    {
        "slice_ops.cl",
#include "./cl_kernels/slice_ops.clembed"
    },
    {
        "space_to_batch.cl",
#include "./cl_kernels/space_to_batch.clembed"
    },
    {
        "space_to_depth.cl",
#include "./cl_kernels/space_to_depth.clembed"
    },
    {
        "stack_layer.cl",
#include "./cl_kernels/stack_layer.clembed"
    },
    {
        "tile.cl",
#include "./cl_kernels/tile.clembed"
    },
    {
        "transpose.cl",
#include "./cl_kernels/transpose.clembed"
    },
    {
        "types.h",
#include "./cl_kernels/types.hembed"
    },
    {
        "unpooling_layer.cl",
#include "./cl_kernels/unpooling_layer.clembed"
    },
    {
        "winograd_filter_transform.cl",
#include "./cl_kernels/winograd_filter_transform.clembed"
    },
    {
        "winograd_input_transform.cl",
#include "./cl_kernels/winograd_input_transform.clembed"
    },
    {
        "winograd_output_transform.cl",
#include "./cl_kernels/winograd_output_transform.clembed"
    },
    {
        "yolo_layer.cl",
#include "./cl_kernels/yolo_layer.clembed"
    },
#endif /* EMBEDDED_KERNELS */
};

CLKernelLibrary::CLKernelLibrary()
    : _compile_context(), _kernel_path(), _decompressed_source_map()
{
    opencl_is_available(); // Make sure the OpenCL symbols are initialised *before* the CLKernelLibrary is built
}

CLKernelLibrary &CLKernelLibrary::get()
{
    static CLKernelLibrary _kernel_library;
    return _kernel_library;
}

Kernel CLKernelLibrary::create_kernel(const std::string &kernel_name, const std::set<std::string> &build_options_set) const
{
    const std::string program_name = get_program_name(kernel_name);
    auto              program      = get_program(program_name);

    return _compile_context.create_kernel(kernel_name, program_name, program.first, _kernel_path, build_options_set, program.second);
}

std::string CLKernelLibrary::get_program_name(const std::string &kernel_name) const
{
    // Find which program contains the kernel
    auto kernel_program_it = _kernel_program_map.find(kernel_name);

    if(_kernel_program_map.end() == kernel_program_it)
    {
        ARM_COMPUTE_ERROR_VAR("Kernel %s not found in the CLKernelLibrary", kernel_name.c_str());
    }

    const std::string program_name = kernel_program_it->second;

    return program_name;
}

void CLKernelLibrary::init(std::string kernel_path, cl::Context context, cl::Device device)
{
    _compile_context = CLCompileContext(context, device);
    _kernel_path     = kernel_path + "/";
}

void CLKernelLibrary::set_kernel_path(const std::string &kernel_path)
{
    _kernel_path = std::move(kernel_path);
    _kernel_path += "/";
}

cl::Context &CLKernelLibrary::context()
{
    return _compile_context.context();
}

const cl::Device &CLKernelLibrary::get_device()
{
    return _compile_context.get_device();
}

void CLKernelLibrary::set_device(cl::Device device)
{
    _compile_context.set_device(device);
}

void CLKernelLibrary::set_context(cl::Context context)
{
    _compile_context.set_context(context);
}

std::string CLKernelLibrary::get_kernel_path()
{
    return _kernel_path;
}

void CLKernelLibrary::clear_programs_cache()
{
    _compile_context.clear_programs_cache();
}

const std::map<std::string, cl::Program> &CLKernelLibrary::get_built_programs() const
{
    return _compile_context.get_built_programs();
}

void CLKernelLibrary::add_built_program(const std::string &built_program_name, const cl::Program &program)
{
    _compile_context.add_built_program(built_program_name, program);
}

bool CLKernelLibrary::fp16_supported() const
{
    return _compile_context.fp16_supported();
}

bool CLKernelLibrary::int64_base_atomics_supported() const
{
    return _compile_context.int64_base_atomics_supported();
}

bool CLKernelLibrary::is_wbsm_supported()
{
    return _compile_context.is_wbsm_supported();
}

std::pair<std::string, bool> CLKernelLibrary::get_program(const std::string &program_name) const
{
#ifdef EMBEDDED_KERNELS
#ifdef ARM_COMPUTE_COMPRESSED_KERNELS
    const auto inflatted_program_source_it = _decompressed_source_map.find(program_name);
    if(inflatted_program_source_it != _decompressed_source_map.end())
    {
        return std::make_pair(inflatted_program_source_it->second, false);
    }
#endif /* ARM_COMPUTE_COMPRESSED_KERNELS */

    const auto program_source_it = _program_source_map.find(program_name);
    if(program_source_it == _program_source_map.end())
    {
        ARM_COMPUTE_ERROR_VAR("Embedded program for %s does not exist.", program_name.c_str());
    }
    std::string program_source = program_source_it->second;

#ifdef ARM_COMPUTE_COMPRESSED_KERNELS
    std::string decompressed_program_source = decompress_zlib(decode_base64(program_source_it->second));
    ARM_COMPUTE_ERROR_ON_MSG(decompressed_program_source.empty(), "Cannot de-compress requested program");
    _decompressed_source_map.insert(std::make_pair(program_name, decompressed_program_source));
    program_source = std::move(decompressed_program_source);
#endif /* ARM_COMPUTE_COMPRESSED_KERNELS */

    return std::make_pair(program_source, false);
#else  /* EMBEDDED_KERNELS */
    // Check for binary
    std::string source_name = _kernel_path + program_name;
    std::string binary_name = source_name + "bin";
    std::string program_source{};
    bool        is_binary = false;

    if(std::ifstream(binary_name).is_open())
    {
        program_source = read_file(binary_name, true);
        is_binary      = true;
    }
    else if(std::ifstream(source_name).is_open())
    {
        program_source = read_file(source_name, false);
    }
    else
    {
        ARM_COMPUTE_ERROR_VAR("Kernel file %s does not exist.", source_name.c_str());
    }

    return std::make_pair(program_source, is_binary);
#endif /* EMBEDDED_KERNELS */
}

size_t CLKernelLibrary::max_local_workgroup_size(const cl::Kernel &kernel) const
{
    return _compile_context.max_local_workgroup_size(kernel);
}

cl::NDRange CLKernelLibrary::default_ndrange() const
{
    return _compile_context.default_ndrange();
}

std::string CLKernelLibrary::get_device_version()
{
    return _compile_context.get_device_version();
}

cl_uint CLKernelLibrary::get_num_compute_units()
{
    return _compile_context.get_num_compute_units();
}

CLCompileContext &CLKernelLibrary::get_compile_context()
{
    return _compile_context;
}
