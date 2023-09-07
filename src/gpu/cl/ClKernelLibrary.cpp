/*
 * Copyright (c) 2016-2023 Arm Limited.
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
#include "src/gpu/cl/ClKernelLibrary.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Utils.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <utility>

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

namespace arm_compute
{
namespace opencl
{
const std::map<std::string, std::string> ClKernelLibrary::_kernel_program_map =
{
    // Common Kernels
    { "activation_layer", "common/activation_layer.cl" },
    { "activation_layer_quant", "common/activation_layer_quant.cl" },
    { "activation_layer_quant_f32", "common/activation_layer_quant.cl" },
    { "arg_min_max_x", "common/arg_min_max.cl" },
    { "arg_min_max_y", "common/arg_min_max.cl" },
    { "arg_min_max_z", "common/arg_min_max.cl" },
    { "arg_min_max_w", "common/arg_min_max.cl" },
    { "bitwise_or", "common/bitwise_op.cl" },
    { "bitwise_and", "common/bitwise_op.cl" },
    { "bitwise_xor", "common/bitwise_op.cl" },
    { "bitwise_not", "common/bitwise_op.cl" },
    { "bounding_box_transform", "common/bounding_box_transform.cl" },
    { "bounding_box_transform_quantized", "common/bounding_box_transform_quantized.cl" },
    { "compare_equal", "common/comparisons.cl" },
    { "compare_equal_quantized", "common/comparisons.cl" },
    { "compare_notequal", "common/comparisons.cl" },
    { "compare_notequal_quantized", "common/comparisons.cl" },
    { "compare_greater", "common/comparisons.cl" },
    { "compare_greater_quantized", "common/comparisons.cl" },
    { "compare_greaterequal", "common/comparisons.cl" },
    { "compare_greaterequal_quantized", "common/comparisons.cl" },
    { "compare_less", "common/comparisons.cl" },
    { "compare_less_quantized", "common/comparisons.cl" },
    { "compare_lessequal", "common/comparisons.cl" },
    { "compare_lessequal_quantized", "common/comparisons.cl" },
    { "concatenate", "common/concatenate.cl" },
    { "concatenate_width", "common/concatenate.cl" },
    { "concatenate_height", "common/concatenate.cl" },
    { "concatenate_width_x2", "common/concatenate.cl" },
    { "concatenate_width_x4", "common/concatenate.cl" },
    { "col2im", "common/col2im.cl" },
    { "cast_down", "common/cast.cl" },
    { "cast_up", "common/cast.cl" },
    { "convert_fc_weights", "common/convert_fc_weights.cl" },
    { "copy_tensor", "common/copy_tensor.cl" },
    { "crop_tensor", "common/crop_tensor.cl" },
    { "deconvolution_reshape", "common/deconvolution_layer.cl" },
    { "deconvolution_upsample", "common/deconvolution_layer.cl" },
    { "dequantization_layer", "common/dequantization_layer.cl" },
    { "elementwise_operation_ADD", "common/elementwise_operation.cl" },
    { "elementwise_operation_SUB", "common/elementwise_operation.cl" },
    { "elementwise_operation_MAX", "common/elementwise_operation.cl" },
    { "elementwise_operation_MIN", "common/elementwise_operation.cl" },
    { "elementwise_operation_DIV", "common/elementwise_operation.cl" },
    { "elementwise_operation_SQUARED_DIFF", "common/elementwise_operation.cl" },
    { "elementwise_operation_POWER", "common/elementwise_operation.cl" },
    { "elementwise_operation_PRELU", "common/elementwise_operation.cl" },
    { "elementwise_operation_AND", "common/elementwise_operation.cl" },
    { "elementwise_operation_OR", "common/elementwise_operation.cl" },
    { "elementwise_operation_ADD_quantized", "common/elementwise_operation_quantized.cl" },
    { "elementwise_operation_SUB_quantized", "common/elementwise_operation_quantized.cl" },
    { "elementwise_operation_MAX_quantized", "common/elementwise_operation_quantized.cl" },
    { "elementwise_operation_MIN_quantized", "common/elementwise_operation_quantized.cl" },
    { "elementwise_operation_DIV_quantized", "common/elementwise_operation_quantized.cl" },
    { "elementwise_operation_SQUARED_DIFF_quantized", "common/elementwise_operation_quantized.cl" },
    { "elementwise_operation_PRELU_quantized", "common/elementwise_operation_quantized.cl" },
    { "elementwise_unary", "common/elementwise_unary.cl" },
    { "elementwise_unary_quantized", "common/elementwise_unary_quantized.cl" },
    { "fft_digit_reverse_axis_0", "common/fft_digit_reverse.cl" },
    { "fft_digit_reverse_axis_1", "common/fft_digit_reverse.cl" },
    { "fft_radix_2_first_stage_axis_0", "common/fft.cl" },
    { "fft_radix_2_first_stage_axis_1", "common/fft.cl" },
    { "fft_radix_2_axis_0", "common/fft.cl" },
    { "fft_radix_2_axis_1", "common/fft.cl" },
    { "fft_radix_3_first_stage_axis_0", "common/fft.cl" },
    { "fft_radix_3_first_stage_axis_1", "common/fft.cl" },
    { "fft_radix_3_axis_0", "common/fft.cl" },
    { "fft_radix_3_axis_1", "common/fft.cl" },
    { "fft_radix_4_first_stage_axis_0", "common/fft.cl" },
    { "fft_radix_4_first_stage_axis_1", "common/fft.cl" },
    { "fft_radix_4_axis_0", "common/fft.cl" },
    { "fft_radix_4_axis_1", "common/fft.cl" },
    { "fft_radix_5_first_stage_axis_0", "common/fft.cl" },
    { "fft_radix_5_first_stage_axis_1", "common/fft.cl" },
    { "fft_radix_5_axis_0", "common/fft.cl" },
    { "fft_radix_5_axis_1", "common/fft.cl" },
    { "fft_radix_7_first_stage_axis_0", "common/fft.cl" },
    { "fft_radix_7_first_stage_axis_1", "common/fft.cl" },
    { "fft_radix_7_axis_0", "common/fft.cl" },
    { "fft_radix_7_axis_1", "common/fft.cl" },
    { "fft_radix_8_first_stage_axis_0", "common/fft.cl" },
    { "fft_radix_8_first_stage_axis_1", "common/fft.cl" },
    { "fft_radix_8_axis_0", "common/fft.cl" },
    { "fft_radix_8_axis_1", "common/fft.cl" },
    { "fft_scale_conj", "common/fft_scale.cl" },
    { "fill_image_borders_constant", "common/fill_border.cl" },
    { "fill_image_borders_replicate", "common/fill_border.cl" },
    { "floor_layer", "common/floor.cl" },
    { "fuse_batchnormalization_layer", "common/batchnormalization_layer.cl" },
    { "gather", "common/gather.cl" },
    { "gemm_ma_f16", "common/gemm.cl" },
    { "gemm_ma_f32", "common/gemm.cl" },
    { "gemm_mv", "common/gemv.cl" },
    { "gemm_mv_quantized", "common/gemv.cl" },
    { "gemm_mm_native", "common/gemm.cl" },
    { "gemm_mm_reshaped_only_rhs_nt_mmul", "common/gemm_reshaped_only_rhs_mmul.cl" },
    { "gemm_mm_reshaped_only_rhs_nt_mmul_texture", "common/gemm_reshaped_only_rhs_mmul.cl" },
    { "gemm_mm_reshaped_lhs_nt_rhs_t", "common/gemm.cl" },
    { "gemm_mm_reshaped_lhs_nt_rhs_t_texture", "common/gemm.cl" },
    { "gemm_mm_reshaped_lhs_t_rhs_nt", "common/gemm.cl" },
    { "gemm_mm_reshaped_lhs_t_rhs_nt_texture", "common/gemm.cl" },
    { "gemm_mm_reshaped_only_rhs_nt", "common/gemm.cl" },
    { "gemm_mm_reshaped_only_rhs_nt_texture", "common/gemm.cl" },
    { "gemm_mm_reshaped_only_rhs_t", "common/gemm.cl" },
    { "gemm_mm_reshaped_only_rhs_t_texture", "common/gemm.cl" },
    { "gemm_lc_vm_f32", "common/gemm.cl" },
    { "gemm_reshape_lhs_matrix_nt", "common/gemm_utils.cl" },
    { "gemm_reshape_lhs_matrix_t", "common/gemm_utils.cl" },
    { "gemm_reshape_rhs_matrix_nt", "common/gemm_utils.cl" },
    { "gemm_reshape_rhs_matrix_t", "common/gemm_utils.cl" },
    { "gemmlowp_matrix_a_reduction", "common/gemmlowp.cl" },
    { "gemmlowp_matrix_a_reduction_dot8", "common/gemmlowp.cl" },
    { "gemmlowp_matrix_b_reduction", "common/gemmlowp.cl" },
    { "gemmlowp_mm_native", "common/gemmlowp.cl" },
    { "gemmlowp_mm_reshaped_lhs_nt_rhs_t", "common/gemmlowp.cl" },
    { "gemmlowp_mm_reshaped_only_rhs_t", "common/gemmlowp.cl" },
    { "gemmlowp_mm_reshaped_only_rhs_t_fused_output_stage_fixedpoint", "common/gemmlowp.cl" },
    { "gemmlowp_mm_reshaped_only_rhs_mmul", "common/gemmlowp_reshaped_only_rhs_mmul.cl" },
    { "gemmlowp_offset_contribution", "common/gemmlowp.cl" },
    { "gemmlowp_offset_contribution_quantize_down", "common/gemmlowp.cl" },
    { "gemmlowp_offset_contribution_quantize_down_fixedpoint", "common/gemmlowp.cl" },
    { "gemmlowp_output_stage_quantize_down", "common/gemmlowp.cl" },
    { "gemmlowp_output_stage_quantize_down_fixedpoint", "common/gemmlowp.cl" },
    { "gemmlowp_output_stage_quantize_down_fixedpoint_qsymm16", "common/gemmlowp.cl" },
    { "gemmlowp_output_stage_quantize_down_float", "common/gemmlowp.cl" },
    { "generate_proposals_compute_all_anchors", "common/generate_proposals.cl" },
    { "generate_proposals_compute_all_anchors_quantized", "common/generate_proposals_quantized.cl" },
    { "instance_normalization", "common/instance_normalization.cl" },
    { "compute_mean_var", "common/instance_normalization.cl" },
    { "l2_normalize_x", "common/l2_normalize.cl" },
    { "l2_normalize_y", "common/l2_normalize.cl" },
    { "l2_normalize_z", "common/l2_normalize.cl" },
    { "mat_mul_native_mmul_nt_nt", "common/mat_mul_mmul.cl" },
    { "mat_mul_native_mmul_t_nt", "common/mat_mul_mmul.cl" },
    { "mat_mul_native_mmul_nt_t", "common/mat_mul_mmul.cl" },
    { "mat_mul_native_mmul_t_t", "common/mat_mul_mmul.cl" },
    { "mat_mul_native_nt_nt", "common/mat_mul.cl" },
    { "mat_mul_native_nt_t", "common/mat_mul.cl" },
    { "mat_mul_native_t_nt", "common/mat_mul.cl" },
    { "mat_mul_native_t_t", "common/mat_mul.cl" },
    { "mat_mul_native_quantized_nt_nt", "common/mat_mul_quantized.cl" },
    { "mat_mul_native_quantized_nt_t", "common/mat_mul_quantized.cl" },
    { "mat_mul_native_quantized_t_nt", "common/mat_mul_quantized.cl" },
    { "mat_mul_native_quantized_t_t", "common/mat_mul_quantized.cl" },
    { "mat_mul_native_quantized_mmul_nt_nt", "common/mat_mul_quantized_mmul.cl" },
    { "mat_mul_native_quantized_mmul_nt_t", "common/mat_mul_quantized_mmul.cl" },
    { "mat_mul_native_quantized_mmul_t_nt", "common/mat_mul_quantized_mmul.cl" },
    { "mat_mul_native_quantized_mmul_t_t", "common/mat_mul_quantized_mmul.cl" },
    { "max_unpooling_layer_2", "common/unpooling_layer.cl" },
    { "mean_stddev_normalization", "common/mean_stddev_normalization.cl" },
    { "memset", "common/memset.cl" },
    { "minmax_layer", "common/minmax_layer.cl" },
    { "non_max_suppression", "common/nonmax.cl" },
    { "pad_layer_constant", "common/pad_layer.cl" },
    { "pad_layer_symmetric_reflect", "common/pad_layer.cl" },
    { "permute", "common/permute.cl" },
    { "pixelwise_mul_complex", "common/pixelwise_mul_float.cl" },
    { "pixelwise_mul_float", "common/pixelwise_mul_float.cl" },
    { "pixelwise_mul_int", "common/pixelwise_mul_int.cl" },
    { "pixelwise_mul_quantized", "common/pixelwise_mul_int.cl" },
    { "qlstm_layer_normalization", "common/qlstm_layer_normalization.cl" },
    { "quantization_layer", "common/quantization_layer.cl" },
    { "range", "common/range.cl" },
    { "range_quantized", "common/range.cl" },
    { "reduction_operation_x", "common/reduction_operation.cl" },
    { "reduction_operation_non_parallel_x", "common/reduction_operation.cl" },
    { "reduction_operation_y", "common/reduction_operation.cl" },
    { "reduction_operation_z", "common/reduction_operation.cl" },
    { "reduction_operation_w", "common/reduction_operation.cl" },
    { "reshape_layer", "common/reshape_layer.cl" },
    { "reshape_to_columns", "common/convolution_layer.cl" },
    { "reverse", "common/reverse.cl" },
    { "roi_align_layer", "common/roi_align_layer.cl" },
    { "roi_align_layer_quantized", "common/roi_align_layer_quantized.cl" },
    { "roi_pooling_layer", "common/roi_pooling_layer.cl" },
    { "select_same_rank", "common/select.cl" },
    { "select_different_rank_2", "common/select.cl" },
    { "select_different_rank_n", "common/select.cl" },
    { "softmax_layer_norm", "common/softmax_layer.cl" },
    { "softmax_layer_norm_quantized", "common/softmax_layer_quantized.cl" },
    { "softmax_layer_max_shift_exp_sum_quantized_serial", "common/softmax_layer_quantized.cl" },
    { "softmax_layer_max_shift_exp_sum_quantized_parallel", "common/softmax_layer_quantized.cl" },
    { "softmax_layer_max_shift_exp_sum_serial", "common/softmax_layer.cl" },
    { "softmax_layer_max_shift_exp_sum_parallel", "common/softmax_layer.cl" },
    { "stack_layer", "common/stack_layer.cl" },
    { "strided_slice", "common/slice_ops.cl" },
    { "tile", "common/tile.cl" },
    { "transpose", "common/transpose.cl" },
#ifdef ENABLE_NCHW_KERNELS
    { "batch_to_space_nchw", "nchw/batch_to_space.cl" },
    { "batch_to_space_static_nchw", "nchw/batch_to_space.cl" },
    { "batchnormalization_layer_nchw", "nchw/batchnormalization_layer.cl" },
    { "channel_shuffle_nchw", "nchw/channel_shuffle.cl" },
    { "depth_to_space_nchw", "nchw/depth_to_space.cl" },
    { "dequantization_layer_per_channel_nchw", "nchw/dequantization_layer.cl" },
    { "direct_convolution1x1", "nchw/direct_convolution1x1.cl" },
    { "direct_convolution_nchw", "nchw/direct_convolution.cl" },

    { "im2col1x1_stridex1_nchw", "nchw/im2col.cl" },
    { "im2col3x3_nchw", "nchw/im2col.cl" },
    { "im2col5x5_nchw", "nchw/im2col.cl" },
    { "im2col11x11_padx0_pady0_nchw", "nchw/im2col.cl" },
    { "im2col_generic_nchw", "nchw/im2col.cl" },
    { "im2col_generic_padx0_pady0_nchw", "nchw/im2col.cl" },
    { "normalization_layer_cross_map_nchw", "nchw/normalization_layer.cl" },
    { "normalization_layer_in_map_nchw", "nchw/normalization_layer.cl" },
    { "normalize_planar_yuv_layer_nchw", "nchw/normalize_planar_yuv_layer.cl" },
    { "normalize_planar_yuv_layer_q8_nchw", "nchw/normalize_planar_yuv_layer_quantized.cl" },
    { "pooling_layer_MxN_nchw", "nchw/pooling_layer.cl" },
    { "pooling_layer_2_nchw_indices", "nchw/pooling_layer.cl" },
    { "prior_box_layer_nchw", "nchw/prior_box_layer.cl" },
    { "reorg_layer_nchw", "nchw/reorg_layer.cl" },
    { "scale_nearest_neighbour_nchw", "nchw/scale.cl" },
    { "scale_bilinear_nchw", "nchw/scale.cl" },
    { "space_to_batch_nchw", "nchw/space_to_batch.cl" },
    { "space_to_batch_static_nchw", "nchw/space_to_batch.cl" },
    { "space_to_depth_nchw", "nchw/space_to_depth.cl" },
    { "upsample_layer_nchw", "nchw/upsample_layer.cl" },
    { "winograd_filter_transform_2x2_3x3_nchw", "nchw/winograd_filter_transform.cl" },
    { "winograd_filter_transform_2x1_3x1_nchw", "nchw/winograd_filter_transform.cl" },
    { "winograd_filter_transform_1x2_1x3_nchw", "nchw/winograd_filter_transform.cl" },
    { "winograd_filter_transform_4x4_3x3_nchw", "nchw/winograd_filter_transform.cl" },
    { "winograd_filter_transform_4x1_3x1_nchw", "nchw/winograd_filter_transform.cl" },
    { "winograd_filter_transform_1x4_1x3_nchw", "nchw/winograd_filter_transform.cl" },
    { "winograd_filter_transform_4x4_5x5_nchw", "nchw/winograd_filter_transform.cl" },
    { "winograd_filter_transform_4x1_5x1_nchw", "nchw/winograd_filter_transform.cl" },
    { "winograd_filter_transform_1x4_1x5_nchw", "nchw/winograd_filter_transform.cl" },
    { "winograd_input_transform_2x2_3x3_stepz1_nchw", "nchw/winograd_input_transform.cl" },
    { "winograd_input_transform_2x2_3x3_stepz2_nchw", "nchw/winograd_input_transform.cl" },
    { "winograd_input_transform_2x1_3x1_stepz1_nchw", "nchw/winograd_input_transform.cl" },
    { "winograd_input_transform_2x1_3x1_stepz2_nchw", "nchw/winograd_input_transform.cl" },
    { "winograd_input_transform_1x2_1x3_stepz1_nchw", "nchw/winograd_input_transform.cl" },
    { "winograd_input_transform_1x2_1x3_stepz2_nchw", "nchw/winograd_input_transform.cl" },
    { "winograd_input_transform_4x4_3x3_stepz1_nchw", "nchw/winograd_input_transform.cl" },
    { "winograd_input_transform_4x1_3x1_stepz1_nchw", "nchw/winograd_input_transform.cl" },
    { "winograd_input_transform_1x4_1x3_stepz1_nchw", "nchw/winograd_input_transform.cl" },
    { "winograd_input_transform_4x4_5x5_stepz1_nchw", "nchw/winograd_input_transform.cl" },
    { "winograd_input_transform_4x1_5x1_stepz1_nchw", "nchw/winograd_input_transform.cl" },
    { "winograd_input_transform_1x4_1x5_stepz1_nchw", "nchw/winograd_input_transform.cl" },
    { "winograd_output_transform_2x2_3x3_nchw", "nchw/winograd_output_transform.cl" },
    { "winograd_output_transform_2x1_3x1_nchw", "nchw/winograd_output_transform.cl" },
    { "winograd_output_transform_1x2_1x3_nchw", "nchw/winograd_output_transform.cl" },
    { "winograd_output_transform_4x4_3x3_nchw", "nchw/winograd_output_transform.cl" },
    { "winograd_output_transform_4x1_3x1_nchw", "nchw/winograd_output_transform.cl" },
    { "winograd_output_transform_1x4_1x3_nchw", "nchw/winograd_output_transform.cl" },
    { "winograd_output_transform_4x4_5x5_nchw", "nchw/winograd_output_transform.cl" },
    { "winograd_output_transform_4x1_5x1_nchw", "nchw/winograd_output_transform.cl" },
    { "winograd_output_transform_1x4_1x5_nchw", "nchw/winograd_output_transform.cl" },
#endif /* ENABLE_NCHW_KERNELS */
#ifdef ENABLE_NHWC_KERNELS
    { "batch_to_space_nhwc", "nhwc/batch_to_space.cl" },
    { "batch_to_space_static_nhwc", "nhwc/batch_to_space.cl" },
    { "batchnormalization_layer_nhwc", "nhwc/batchnormalization_layer.cl" },
    { "channel_shuffle_nhwc", "nhwc/channel_shuffle.cl" },
    { "depth_to_space_nhwc", "nhwc/depth_to_space.cl" },
    { "dequantization_layer_per_channel_nhwc", "nhwc/dequantization_layer.cl" },
    { "dwc_native_fp_nhwc", "nhwc/dwc_native_fp_nhwc.cl" },
    { "dwc_native_quantized_nhwc", "nhwc/dwc_native_quantized_nhwc.cl" },
    { "direct_convolution_nhwc", "nhwc/direct_convolution.cl" },
    { "direct_convolution3d_ndhwc", "nhwc/direct_convolution3d.cl" },
    { "im2col3x3_nhwc", "nhwc/im2col.cl" },
    { "im2col9x9_nhwc", "nhwc/im2col.cl" },
    { "im2col_generic_nhwc", "nhwc/im2col.cl" },
    { "indirect_convolution_nhwc", "nhwc/indirect_convolution.cl" },
    { "indirect_convolution_address_precalculation", "nhwc/indirect_convolution.cl" },
    { "normalization_layer_cross_map_nhwc", "nhwc/normalization_layer.cl" },
    { "normalization_layer_in_map_nhwc", "nhwc/normalization_layer.cl" },
    { "normalize_planar_yuv_layer_nhwc", "nhwc/normalize_planar_yuv_layer.cl" },
    { "normalize_planar_yuv_layer_q8_nhwc", "nhwc/normalize_planar_yuv_layer_quantized.cl" },
    { "pooling_layer_MxN_nhwc", "nhwc/pooling_layer.cl" },
    { "pooling_layer_2x2_nhwc", "nhwc/pooling_layer.cl" },
    { "pooling_layer_MxN_quantized_nhwc", "nhwc/pooling_layer_quantized.cl" },
    { "pooling_3d_layer_MxN_ndhwc", "nhwc/pooling_3d_layer.cl" },
    { "pooling_3d_layer_MxN_ndhwc_quantized", "nhwc/pooling_3d_layer_quantized.cl" },
    { "reorg_layer_nhwc", "nhwc/reorg_layer.cl" },
    { "scale_nearest_neighbour_nhwc", "nhwc/scale.cl" },
    { "scale_bilinear_nhwc", "nhwc/scale.cl" },
    { "space_to_batch_nhwc", "nhwc/space_to_batch.cl" },
    { "space_to_batch_static_nhwc", "nhwc/space_to_batch.cl" },
    { "space_to_depth_nhwc", "nhwc/space_to_depth.cl" },
    { "transposed_convolution_nhwc", "nhwc/transposed_convolution.cl" },
    { "upsample_layer_nhwc", "nhwc/upsample_layer.cl" },
    { "winograd_filter_transform_4x1_3x1_nhwc", "nhwc/winograd_filter_transform.cl" },
    { "winograd_filter_transform_1x4_1x3_nhwc", "nhwc/winograd_filter_transform.cl" },
    { "winograd_filter_transform_4x4_3x3_nhwc", "nhwc/winograd_filter_transform.cl" },
    { "winograd_filter_transform_4x4_5x5_nhwc", "nhwc/winograd_filter_transform.cl" },
    { "winograd_filter_transform_4x1_5x1_nhwc", "nhwc/winograd_filter_transform.cl" },
    { "winograd_filter_transform_1x4_1x5_nhwc", "nhwc/winograd_filter_transform.cl" },
    { "winograd_filter_transform_2x2_7x7_nhwc", "nhwc/winograd_filter_transform.cl" },
    { "winograd_filter_transform_2x1_7x1_nhwc", "nhwc/winograd_filter_transform.cl" },
    { "winograd_filter_transform_1x2_1x7_nhwc", "nhwc/winograd_filter_transform.cl" },
    { "winograd_input_transform_4x1_3x1_stepz1_nhwc", "nhwc/winograd_input_transform.cl" },
    { "winograd_input_transform_1x4_1x3_stepz1_nhwc", "nhwc/winograd_input_transform.cl" },
    { "winograd_input_transform_4x4_3x3_stepz1_nhwc", "nhwc/winograd_input_transform.cl" },
    { "winograd_input_transform_4x4_5x5_stepz1_nhwc", "nhwc/winograd_input_transform.cl" },
    { "winograd_input_transform_4x1_5x1_stepz1_nhwc", "nhwc/winograd_input_transform.cl" },
    { "winograd_input_transform_1x4_1x5_stepz1_nhwc", "nhwc/winograd_input_transform.cl" },
    { "winograd_input_transform_2x2_7x7_stepz1_nhwc", "nhwc/winograd_input_transform.cl" },
    { "winograd_input_transform_2x1_7x1_stepz1_nhwc", "nhwc/winograd_input_transform.cl" },
    { "winograd_input_transform_1x2_1x7_stepz1_nhwc", "nhwc/winograd_input_transform.cl" },
    { "winograd_output_transform_4x1_3x1_nhwc", "nhwc/winograd_output_transform.cl" },
    { "winograd_output_transform_1x4_1x3_nhwc", "nhwc/winograd_output_transform.cl" },
    { "winograd_output_transform_4x4_3x3_nhwc", "nhwc/winograd_output_transform.cl" },
    { "winograd_output_transform_4x4_5x5_nhwc", "nhwc/winograd_output_transform.cl" },
    { "winograd_output_transform_4x1_5x1_nhwc", "nhwc/winograd_output_transform.cl" },
    { "winograd_output_transform_1x4_1x5_nhwc", "nhwc/winograd_output_transform.cl" },
    { "winograd_output_transform_2x2_7x7_nhwc", "nhwc/winograd_output_transform.cl" },
    { "winograd_output_transform_2x1_7x1_nhwc", "nhwc/winograd_output_transform.cl" },
    { "winograd_output_transform_1x2_1x7_nhwc", "nhwc/winograd_output_transform.cl" },
#endif /* ENABLE_NHWC_KERNELS */
};

const std::map<std::string, std::string> ClKernelLibrary::_program_source_map =
{
#ifdef EMBEDDED_KERNELS
    {
        "activation_float_helpers.h",
#include "./cl_kernels/activation_float_helpers.hembed"
    },
    {
        "activation_quant_helpers.h",
#include "./cl_kernels/activation_quant_helpers.hembed"
    },
    {
        "common/activation_layer.cl",
#include "./cl_kernels/common/activation_layer.clembed"
    },
    {
        "common/activation_layer_quant.cl",
#include "./cl_kernels/common/activation_layer_quant.clembed"
    },
    {
        "common/arg_min_max.cl",
#include "./cl_kernels/common/arg_min_max.clembed"
    },
    {
        "common/bitwise_op.cl",
#include "./cl_kernels/common/bitwise_op.clembed"
    },
    {
        "common/bounding_box_transform.cl",
#include "./cl_kernels/common/bounding_box_transform.clembed"
    },
    {
        "common/bounding_box_transform_quantized.cl",
#include "./cl_kernels/common/bounding_box_transform_quantized.clembed"
    },
    {
        "common/col2im.cl",
#include "./cl_kernels/common/col2im.clembed"
    },
    {
        "common/comparisons.cl",
#include "./cl_kernels/common/comparisons.clembed"
    },
    {
        "common/concatenate.cl",
#include "./cl_kernels/common/concatenate.clembed"
    },
    {
        "common/convert_fc_weights.cl",
#include "./cl_kernels/common/convert_fc_weights.clembed"
    },
    {
        "common/convolution_layer.cl",
#include "./cl_kernels/common/convolution_layer.clembed"
    },
    {
        "common/copy_tensor.cl",
#include "./cl_kernels/common/copy_tensor.clembed"
    },
    {
        "common/crop_tensor.cl",
#include "./cl_kernels/common/crop_tensor.clembed"
    },
    {
        "common/deconvolution_layer.cl",
#include "./cl_kernels/common/deconvolution_layer.clembed"
    },
    {
        "common/cast.cl",
#include "./cl_kernels/common/cast.clembed"
    },
    {
        "common/dequantization_layer.cl",
#include "./cl_kernels/common/dequantization_layer.clembed"
    },
    {
        "common/elementwise_operation.cl",
#include "./cl_kernels/common/elementwise_operation.clembed"
    },
    {
        "common/elementwise_operation_quantized.cl",
#include "./cl_kernels/common/elementwise_operation_quantized.clembed"
    },
    {
        "common/elementwise_unary.cl",
#include "./cl_kernels/common/elementwise_unary.clembed"
    },
    {
        "common/elementwise_unary_quantized.cl",
#include "./cl_kernels/common/elementwise_unary_quantized.clembed"
    },
    {
        "common/fft.cl",
#include "./cl_kernels/common/fft.clembed"
    },
    {
        "common/fft_digit_reverse.cl",
#include "./cl_kernels/common/fft_digit_reverse.clembed"
    },
    {
        "common/fft_scale.cl",
#include "./cl_kernels/common/fft_scale.clembed"
    },
    {
        "common/fill_border.cl",
#include "./cl_kernels/common/fill_border.clembed"
    },
    {
        "common/floor.cl",
#include "./cl_kernels/common/floor.clembed"
    },
    {
        "common/gather.cl",
#include "./cl_kernels/common/gather.clembed"
    },
    {
        "common/gemm.cl",
#include "./cl_kernels/common/gemm.clembed"
    },
    {
        "common/gemm_reshaped_only_rhs_mmul.cl",
#include "./cl_kernels/common/gemm_reshaped_only_rhs_mmul.clembed"
    },
    {
        "common/gemm_utils.cl",
#include "./cl_kernels/common/gemm_utils.clembed"
    },
    {
        "common/gemmlowp.cl",
#include "./cl_kernels/common/gemmlowp.clembed"
    },
    {
        "common/gemmlowp_reshaped_only_rhs_mmul.cl",
#include "./cl_kernels/common/gemmlowp_reshaped_only_rhs_mmul.clembed"
    },
    {
        "common/gemv.cl",
#include "./cl_kernels/common/gemv.clembed"
    },
    {
        "common/generate_proposals.cl",
#include "./cl_kernels/common/generate_proposals.clembed"
    },
    {
        "common/generate_proposals_quantized.cl",
#include "./cl_kernels/common/generate_proposals_quantized.clembed"
    },
    {
        "gemm_helpers.h",
#include "./cl_kernels/gemm_helpers.hembed"
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
        "repeat.h",
#include "./cl_kernels/repeat.hembed"
    },
    {
        "tile_helpers.h",
#include "./cl_kernels/tile_helpers.hembed"
    },
    {
        "common/instance_normalization.cl",
#include "./cl_kernels/common/instance_normalization.clembed"
    },
    {
        "common/l2_normalize.cl",
#include "./cl_kernels/common/l2_normalize.clembed"
    },
    {
        "common/mean_stddev_normalization.cl",
#include "./cl_kernels/common/mean_stddev_normalization.clembed"
    },
    {
        "common/memset.cl",
#include "./cl_kernels/common/memset.clembed"
    },
    {
        "common/minmax_layer.cl",
#include "./cl_kernels/common/minmax_layer.clembed"
    },
    {
        "common/nonmax.cl",
#include "./cl_kernels/common/nonmax.clembed"
    },
    {
        "common/batchnormalization_layer.cl",
#include "./cl_kernels/common/batchnormalization_layer.clembed"
    },
    {
        "common/pad_layer.cl",
#include "./cl_kernels/common/pad_layer.clembed"
    },
    {
        "common/permute.cl",
#include "./cl_kernels/common/permute.clembed"
    },
    {
        "common/pixelwise_mul_float.cl",
#include "./cl_kernels/common/pixelwise_mul_float.clembed"
    },
    {
        "common/pixelwise_mul_int.cl",
#include "./cl_kernels/common/pixelwise_mul_int.clembed"
    },
    {
        "common/qlstm_layer_normalization.cl",
#include "./cl_kernels/common/qlstm_layer_normalization.clembed"
    },
    {
        "common/quantization_layer.cl",
#include "./cl_kernels/common/quantization_layer.clembed"
    },
    {
        "common/range.cl",
#include "./cl_kernels/common/range.clembed"
    },
    {
        "common/reduction_operation.cl",
#include "./cl_kernels/common/reduction_operation.clembed"
    },
    {
        "common/reshape_layer.cl",
#include "./cl_kernels/common/reshape_layer.clembed"
    },
    {
        "common/reverse.cl",
#include "./cl_kernels/common/reverse.clembed"
    },
    {
        "common/roi_align_layer.cl",
#include "./cl_kernels/common/roi_align_layer.clembed"
    },
    {
        "common/roi_align_layer_quantized.cl",
#include "./cl_kernels/common/roi_align_layer_quantized.clembed"
    },
    {
        "common/roi_pooling_layer.cl",
#include "./cl_kernels/common/roi_pooling_layer.clembed"
    },
    {
        "common/select.cl",
#include "./cl_kernels/common/select.clembed"
    },
    {
        "common/softmax_layer.cl",
#include "./cl_kernels/common/softmax_layer.clembed"
    },
    {
        "common/softmax_layer_quantized.cl",
#include "./cl_kernels/common/softmax_layer_quantized.clembed"
    },
    {
        "common/slice_ops.cl",
#include "./cl_kernels/common/slice_ops.clembed"
    },
    {
        "common/stack_layer.cl",
#include "./cl_kernels/common/stack_layer.clembed"
    },
    {
        "common/tile.cl",
#include "./cl_kernels/common/tile.clembed"
    },
    {
        "common/transpose.cl",
#include "./cl_kernels/common/transpose.clembed"
    },
    {
        "types.h",
#include "./cl_kernels/types.hembed"
    },
    {
        "common/unpooling_layer.cl",
#include "./cl_kernels/common/unpooling_layer.clembed"
    },
    {
        "common/mat_mul.cl",
#include "./cl_kernels/common/mat_mul.clembed"
    },
    {
        "common/mat_mul_mmul.cl",
#include "./cl_kernels/common/mat_mul_mmul.clembed"
    },
    {
        "common/mat_mul_quantized.cl",
#include "./cl_kernels/common/mat_mul_quantized.clembed"
    },
    {
        "common/mat_mul_quantized_mmul.cl",
#include "./cl_kernels/common/mat_mul_quantized_mmul.clembed"
    },
#ifdef ENABLE_NCHW_KERNELS
    {
        "nchw/batch_to_space.cl",
#include "./cl_kernels/nchw/batch_to_space.clembed"
    },
    {
        "nchw/channel_shuffle.cl",
#include "./cl_kernels/nchw/channel_shuffle.clembed"
    },
    {
        "nchw/upsample_layer.cl",
#include "./cl_kernels/nchw/upsample_layer.clembed"
    },
    {
        "nchw/depth_to_space.cl",
#include "./cl_kernels/nchw/depth_to_space.clembed"
    },
    {
        "nchw/dequantization_layer.cl",
#include "./cl_kernels/nchw/dequantization_layer.clembed"
    },
    {
        "nchw/direct_convolution.cl",
#include "./cl_kernels/nchw/direct_convolution.clembed"
    },
    {
        "nchw/im2col.cl",
#include "./cl_kernels/nchw/im2col.clembed"
    },
    {
        "nchw/normalization_layer.cl",
#include "./cl_kernels/nchw/normalization_layer.clembed"
    },
    {
        "nchw/normalize_planar_yuv_layer.cl",
#include "./cl_kernels/nchw/normalize_planar_yuv_layer.clembed"
    },
    {
        "nchw/normalize_planar_yuv_layer_quantized.cl",
#include "./cl_kernels/nchw/normalize_planar_yuv_layer_quantized.clembed"
    },
    {
        "nchw/batchnormalization_layer.cl",
#include "./cl_kernels/nchw/batchnormalization_layer.clembed"
    },
    {
        "nchw/pooling_layer.cl",
#include "./cl_kernels/nchw/pooling_layer.clembed"
    },
    {
        "nchw/prior_box_layer.cl",
#include "./cl_kernels/nchw/prior_box_layer.clembed"
    },
    {
        "nchw/reorg_layer.cl",
#include "./cl_kernels/nchw/reorg_layer.clembed"
    },
    {
        "nchw/scale.cl",
#include "./cl_kernels/nchw/scale.clembed"
    },
    {
        "nchw/space_to_batch.cl",
#include "./cl_kernels/nchw/space_to_batch.clembed"
    },
    {
        "nchw/space_to_depth.cl",
#include "./cl_kernels/nchw/space_to_depth.clembed"
    },
    {
        "nchw/winograd_filter_transform.cl",
#include "./cl_kernels/nchw/winograd_filter_transform.clembed"
    },
    {
        "nchw/winograd_input_transform.cl",
#include "./cl_kernels/nchw/winograd_input_transform.clembed"
    },
    {
        "nchw/winograd_output_transform.cl",
#include "./cl_kernels/nchw/winograd_output_transform.clembed"
    },
#endif /* ENABLE_NCHW_KERNELS */

#ifdef ENABLE_NHWC_KERNELS
    {
        "nhwc/batch_to_space.cl",
#include "./cl_kernels/nhwc/batch_to_space.clembed"
    },
    {
        "nhwc/channel_shuffle.cl",
#include "./cl_kernels/nhwc/channel_shuffle.clembed"
    },
    {
        "nhwc/upsample_layer.cl",
#include "./cl_kernels/nhwc/upsample_layer.clembed"
    },
    {
        "nhwc/depth_to_space.cl",
#include "./cl_kernels/nhwc/depth_to_space.clembed"
    },
    {
        "nhwc/dequantization_layer.cl",
#include "./cl_kernels/nhwc/dequantization_layer.clembed"
    },
    {
        "nhwc/direct_convolution.cl",
#include "./cl_kernels/nhwc/direct_convolution.clembed"
    },
    {
        "nhwc/direct_convolution3d.cl",
#include "./cl_kernels/nhwc/direct_convolution3d.clembed"
    },
    {
        "nhwc/dwc_native_fp_nhwc.cl",
#include "./cl_kernels/nhwc/dwc_native_fp_nhwc.clembed"
    },
    {
        "nhwc/dwc_native_quantized_nhwc.cl",
#include "./cl_kernels/nhwc/dwc_native_quantized_nhwc.clembed"
    },
    {
        "nhwc/normalization_layer.cl",
#include "./cl_kernels/nhwc/normalization_layer.clembed"
    },
    {
        "nhwc/normalize_planar_yuv_layer.cl",
#include "./cl_kernels/nhwc/normalize_planar_yuv_layer.clembed"
    },
    {
        "nhwc/normalize_planar_yuv_layer_quantized.cl",
#include "./cl_kernels/nhwc/normalize_planar_yuv_layer_quantized.clembed"
    },
    {
        "nhwc/im2col.cl",
#include "./cl_kernels/nhwc/im2col.clembed"
    },
    {
        "nhwc/indirect_convolution.cl",
#include "./cl_kernels/nhwc/indirect_convolution.clembed"
    },
    {
        "nhwc/batchnormalization_layer.cl",
#include "./cl_kernels/nhwc/batchnormalization_layer.clembed"
    },
    {
        "nhwc/pooling_layer.cl",
#include "./cl_kernels/nhwc/pooling_layer.clembed"
    },
    {
        "nhwc/pooling_3d_layer.cl",
#include "./cl_kernels/nhwc/pooling_3d_layer.clembed"
    },
    {
        "nhwc/pooling_3d_layer_quantized.cl",
#include "./cl_kernels/nhwc/pooling_3d_layer_quantized.clembed"
    },
    {
        "nhwc/pooling_layer_quantized.cl",
#include "./cl_kernels/nhwc/pooling_layer_quantized.clembed"
    },
    {
        "nhwc/reorg_layer.cl",
#include "./cl_kernels/nhwc/reorg_layer.clembed"
    },
    {
        "nhwc/scale.cl",
#include "./cl_kernels/nhwc/scale.clembed"
    },
    {
        "nhwc/space_to_batch.cl",
#include "./cl_kernels/nhwc/space_to_batch.clembed"
    },
    {
        "nhwc/space_to_depth.cl",
#include "./cl_kernels/nhwc/space_to_depth.clembed"
    },
    {
        "nhwc/transposed_convolution.cl",
#include "./cl_kernels/nhwc/transposed_convolution.clembed"
    },
    {
        "nhwc/winograd_filter_transform.cl",
#include "./cl_kernels/nhwc/winograd_filter_transform.clembed"
    },
    {
        "nhwc/winograd_input_transform.cl",
#include "./cl_kernels/nhwc/winograd_input_transform.clembed"
    },
    {
        "nhwc/winograd_output_transform.cl",
#include "./cl_kernels/nhwc/winograd_output_transform.clembed"
    },
#endif /* ENABLE_NHWC_KERNELS */
#endif /* EMBEDDED_KERNELS */
};

ClKernelLibrary &ClKernelLibrary::get()
{
    static ClKernelLibrary _kernel_library;
    return _kernel_library;
}

std::string ClKernelLibrary::program_name(const std::string &kernel_name) const
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

void ClKernelLibrary::set_kernel_path(std::string kernel_path)
{
    _kernel_path = std::move(kernel_path);
    _kernel_path += "/";
}

const std::string &ClKernelLibrary::kernel_path() const
{
    return _kernel_path;
}

ClKernelLibrary::ClProgramInfo ClKernelLibrary::program(const std::string &program_name) const
{
#ifdef EMBEDDED_KERNELS
#ifdef ARM_COMPUTE_COMPRESSED_KERNELS
    const auto inflatted_program_source_it = _decompressed_source_map.find(program_name);
    if(inflatted_program_source_it != _decompressed_source_map.end())
    {
        return ClProgramInfo{ inflatted_program_source_it->second, false };
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

    return ClProgramInfo{ program_source, false };
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

    return ClProgramInfo{ program_source, is_binary };
#endif /* EMBEDDED_KERNELS */
}
} // namespace opencl
} // namespace arm_compute
