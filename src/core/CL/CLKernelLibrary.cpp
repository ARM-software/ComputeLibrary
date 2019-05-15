/*
 * Copyright (c) 2016-2019 ARM Limited.
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

#include <algorithm>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

using namespace arm_compute;

CLBuildOptions::CLBuildOptions()
    : _build_opts()
{
}

void CLBuildOptions::add_option(std::string option)
{
    _build_opts.emplace(std::move(option));
}

void CLBuildOptions::add_option_if(bool cond, std::string option)
{
    if(cond)
    {
        add_option(std::move(option));
    }
}

void CLBuildOptions::add_option_if_else(bool cond, std::string option_true, std::string option_false)
{
    (cond) ? add_option(std::move(option_true)) : add_option(std::move(option_false));
}

void CLBuildOptions::add_options(const StringSet &options)
{
    _build_opts.insert(options.begin(), options.end());
}

void CLBuildOptions::add_options_if(bool cond, const StringSet &options)
{
    if(cond)
    {
        add_options(options);
    }
}

const CLBuildOptions::StringSet &CLBuildOptions::options() const
{
    return _build_opts;
}

Program::Program()
    : _context(), _device(), _is_binary(false), _name(), _source(), _binary()
{
}

Program::Program(cl::Context context, std::string name, std::string source)
    : _context(std::move(context)), _device(), _is_binary(false), _name(std::move(name)), _source(std::move(source)), _binary()
{
}

Program::Program(cl::Context context, cl::Device device, std::string name, std::vector<unsigned char> binary)
    : _context(std::move(context)), _device(std::move(device)), _is_binary(true), _name(std::move(name)), _source(), _binary(std::move(binary))
{
}

Program::operator cl::Program() const
{
    if(_is_binary)
    {
        return cl::Program(_context, { _device }, { _binary });
    }
    else
    {
        return cl::Program(_context, _source, false);
    }
}

bool Program::build(const cl::Program &program, const std::string &build_options)
{
    try
    {
        return program.build(build_options.c_str()) == CL_SUCCESS;
    }
    catch(const cl::Error &e)
    {
        cl_int     err        = CL_SUCCESS;
        const auto build_info = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&err);

        for(auto &pair : build_info)
        {
            std::cerr << pair.second << std::endl;
        }

        return false;
    }
}

cl::Program Program::build(const std::string &build_options) const
{
    cl::Program cl_program = static_cast<cl::Program>(*this);
    build(cl_program, build_options);
    return cl_program;
}

Kernel::Kernel()
    : _name(), _kernel()
{
}

Kernel::Kernel(std::string name, const cl::Program &program)
    : _name(std::move(name)),
      _kernel(cl::Kernel(program, _name.c_str()))
{
}

const std::map<std::string, std::string> CLKernelLibrary::_kernel_program_map =
{
    { "absdiff", "absdiff.cl" },
    { "accumulate", "accumulate.cl" },
    { "accumulate_squared", "accumulate.cl" },
    { "accumulate_weighted", "accumulate.cl" },
    { "activation_layer", "activation_layer.cl" },
    { "activation_layer_qa8", "activation_layer_qa8.cl" },
    { "activation_layer_logistic_qa8", "activation_layer_qa8.cl" },
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
    { "channel_combine_NV", "channel_combine.cl" },
    { "channel_combine_RGB888", "channel_combine.cl" },
    { "channel_combine_RGBA8888", "channel_combine.cl" },
    { "channel_combine_UYVY422", "channel_combine.cl" },
    { "channel_combine_YUYV422", "channel_combine.cl" },
    { "channel_shuffle_nchw", "channel_shuffle.cl" },
    { "channel_shuffle_nhwc", "channel_shuffle.cl" },
    { "channel_extract_NV12", "channel_extract.cl" },
    { "channel_extract_NV21", "channel_extract.cl" },
    { "channel_extract_RGB888", "channel_extract.cl" },
    { "channel_extract_RGBA8888", "channel_extract.cl" },
    { "channel_extract_UYVY422", "channel_extract.cl" },
    { "channel_extract_YUYV422", "channel_extract.cl" },
    { "combine_gradients_L1", "canny.cl" },
    { "combine_gradients_L2", "canny.cl" },
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
    { "concatenate_depth", "concatenate.cl" },
    { "concatenate_width", "concatenate.cl" },
    { "concatenate_height", "concatenate.cl" },
    { "concatenate_width_x2", "concatenate.cl" },
    { "concatenate_width_x4", "concatenate.cl" },
    { "convolution_rectangle", "convolution_rectangle.cl" },
    { "col2im", "col2im.cl" },
    { "convert_depth_down", "depth_convert.cl" },
    { "convert_depth_up", "depth_convert.cl" },
    { "convert_fc_weights", "convert_fc_weights.cl" },
    { "convolution3x3_static", "convolution3x3.cl" },
    { "convolution5x5_static", "convolution5x5.cl" },
    { "convolution7x7_static", "convolution7x7.cl" },
    { "convolution9x9_static", "convolution9x9.cl" },
    { "convolution_separable1x5_static", "convolution5x5.cl" },
    { "convolution_separable5x1_static", "convolution5x5.cl" },
    { "convolution_separable1x7_static", "convolution7x7.cl" },
    { "convolution_separable7x1_static", "convolution7x7.cl" },
    { "convolution_separable1x9_static", "convolution9x9.cl" },
    { "convolution_separable9x1_static", "convolution9x9.cl" },
    { "copy_tensor", "copy_tensor.cl" },
    { "copy_pad_tensor", "copy_tensor.cl" },
    { "copy_plane", "channel_extract.cl" },
    { "copy_planes_3p", "channel_combine.cl" },
    { "copy_to_keypoint", "fast_corners.cl" },
    { "crop_tensor", "crop_tensor.cl" },
    { "deconvolution_reshape", "deconvolution_layer.cl" },
    { "deconvolution_upsample", "deconvolution_layer.cl" },
    { "depthwise_convolution_3x3", "depthwise_convolution.cl" },
    { "depthwise_convolution_3x3_f16", "depthwise_convolution.cl" },
    { "depthwise_convolution_3x3_nhwc", "depthwise_convolution.cl" },
    { "depthwise_convolution_3x3_nhwc_stride1", "depthwise_convolution.cl" },
    { "dwc_3x3_native_qasymm8_nchw", "depthwise_convolution_quantized.cl" },
    { "dwc_3x3_native_qasymm8_dot8_nchw", "depthwise_convolution_quantized.cl" },
    { "dwc_3x3_reshaped_qasymm8_nhwc", "depthwise_convolution_quantized.cl" },
    { "dwc_3x3_reshaped_qasymm8_stride1_nhwc", "depthwise_convolution_quantized.cl" },
    { "dwc_3x3_reshaped_qasymm8_dot8_stride1_nhwc", "depthwise_convolution_quantized.cl" },
    { "depthwise_convolution_3x3_stridex1_stridey1_bifrost_f16", "depthwise_convolution.cl" },
    { "depthwise_convolution_3x3_stridex2_stridey2_bifrost_f16", "depthwise_convolution.cl" },
    { "depthwise_convolution_3x3_stridex1_stridey1_bifrost_f32", "depthwise_convolution.cl" },
    { "depthwise_convolution_3x3_stridex2_stridey2_bifrost_f32", "depthwise_convolution.cl" },
    { "depthwise_convolution_reshape_weights", "depthwise_convolution.cl" },
    { "depthwise_convolution_reshape_weights_generic", "depthwise_convolution.cl" },
    { "depthwise_im2col", "depthwise_convolution.cl" },
    { "depthwise_vector_to_tensor", "depthwise_convolution.cl" },
    { "dequantization_layer", "dequantization_layer.cl" },
    { "derivative", "derivative.cl" },
    { "dilate", "dilate.cl" },
    { "direct_convolution1x1", "direct_convolution1x1.cl" },
    { "direct_convolution1x1_nhwc", "direct_convolution1x1.cl" },
    { "direct_convolution1x1_f32_bifrost", "direct_convolution1x1.cl" },
    { "direct_convolution3x3", "direct_convolution3x3.cl" },
    { "direct_convolution3x3_nhwc", "direct_convolution3x3.cl" },
    { "direct_convolution3x3_f32_bifrost", "direct_convolution3x3.cl" },
    { "direct_convolution5x5", "direct_convolution5x5.cl" },
    { "direct_convolution5x5_nhwc", "direct_convolution5x5.cl" },
    { "direct_convolution5x5_f32_bifrost", "direct_convolution5x5.cl" },
    { "direct_convolution_1x1_3x3_5x5_quantized", "direct_convolution_1x1_3x3_5x5_quantized.cl" },
    { "elementwise_operation_ADD", "elementwise_operation.cl" },
    { "elementwise_operation_SUB", "elementwise_operation.cl" },
    { "elementwise_operation_MAX", "elementwise_operation.cl" },
    { "elementwise_operation_MIN", "elementwise_operation.cl" },
    { "elementwise_operation_DIV", "elementwise_operation.cl" },
    { "elementwise_operation_SQUARED_DIFF", "elementwise_operation.cl" },
    { "elementwise_operation_POWER", "elementwise_operation.cl" },
    { "elementwise_operation_ADD_quantized", "elementwise_operation_quantized.cl" },
    { "elementwise_operation_SUB_quantized", "elementwise_operation_quantized.cl" },
    { "elementwise_operation_MAX_quantized", "elementwise_operation_quantized.cl" },
    { "elementwise_operation_MIN_quantized", "elementwise_operation_quantized.cl" },
    { "elementwise_operation_DIV_quantized", "elementwise_operation_quantized.cl" },
    { "elementwise_operation_SQUARED_DIFF_quantized", "elementwise_operation_quantized.cl" },
    { "elementwise_unary", "elementwise_unary.cl" },
    { "erode", "erode.cl" },
    { "fast_corners", "fast_corners.cl" },
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
    { "finalize", "optical_flow_pyramid_lk.cl" },
    { "flatten", "flatten.cl" },
    { "floor_layer", "floor.cl" },
    { "fuse_batchnormalization_layer", "batchnormalization_layer.cl" },
    { "gather", "gather.cl" },
    { "gaussian1x5_sub_x", "gaussian_pyramid.cl" },
    { "gaussian5x1_sub_y", "gaussian_pyramid.cl" },
    { "gemm_accumulate_biases", "gemm.cl" },
    { "gemm_ma_f16", "gemm.cl" },
    { "gemm_ma_f32", "gemm.cl" },
    { "gemm_mv", "gemv.cl" },
    { "gemm_mv_quantized", "gemv.cl" },
    { "gemm_mm_interleaved_transposed_f16", "gemm.cl" },
    { "gemm_mm_interleaved_transposed_f16_acc32", "gemm.cl" },
    { "gemm_mm_interleaved_transposed_f16_bifrost", "gemm.cl" },
    { "gemm_mm_interleaved_transposed_f32", "gemm.cl" },
    { "gemm_mm_interleaved_transposed_f32_bifrost", "gemm.cl" },
    { "gemm_mm_floating_point", "gemm.cl" },
    { "gemm_mm_floating_point_f16_bifrost", "gemm.cl" },
    { "gemm_mm_floating_point_f16_bifrost_acc32", "gemm.cl" },
    { "gemm_mm_floating_point_f32_bifrost", "gemm.cl" },
    { "gemm_mm_floating_point_f32_bifrost_1000", "gemm.cl" },
    { "gemm_mm_native", "gemm.cl" },
    { "gemm_mm_reshaped_lhs_nt_rhs_t", "gemm.cl" },
    { "gemm_mm_reshaped_only_rhs_nt", "gemm.cl" },
    { "gemm_mm_reshaped_only_rhs_t", "gemm.cl" },
    { "gemm_lc_vm_f32", "gemm.cl" },
    { "gemm_reshape_lhs_matrix_nt", "gemm.cl" },
    { "gemm_reshape_lhs_matrix_t", "gemm.cl" },
    { "gemm_reshape_rhs_matrix_nt", "gemm.cl" },
    { "gemm_reshape_rhs_matrix_t", "gemm.cl" },
    { "gemmlowp_matrix_a_reduction", "gemmlowp.cl" },
    { "gemmlowp_matrix_a_reduction_dot8", "gemmlowp.cl" },
    { "gemmlowp_matrix_b_reduction", "gemmlowp.cl" },
    { "gemmlowp_mm_bifrost", "gemmlowp.cl" },
    { "gemmlowp_mm_bifrost_dot8", "gemmlowp.cl" },
    { "gemmlowp_mm_midgard", "gemmlowp.cl" },
    { "gemmlowp_mm_interleaved_transposed_bifrost", "gemmlowp.cl" },
    { "gemmlowp_mm_interleaved_transposed_bifrost_dot8", "gemmlowp.cl" },
    { "gemmlowp_mm_interleaved_transposed_midgard", "gemmlowp.cl" },
    { "gemmlowp_mm_reshaped_lhs_nt_rhs_t", "gemmlowp.cl" },
    { "gemmlowp_mm_reshaped_lhs_nt_rhs_t_dot8", "gemmlowp.cl" },
    { "gemmlowp_mm_reshaped_only_rhs_t", "gemmlowp.cl" },
    { "gemmlowp_offset_contribution", "gemmlowp.cl" },
    { "gemmlowp_offset_contribution_quantize_down", "gemmlowp.cl" },
    { "gemmlowp_offset_contribution_quantize_down_fixedpoint", "gemmlowp.cl" },
    { "gemmlowp_output_stage_quantize_down", "gemmlowp.cl" },
    { "gemmlowp_output_stage_quantize_down_fixedpoint", "gemmlowp.cl" },
    { "gemmlowp_output_stage_quantize_down_float", "gemmlowp.cl" },
    { "generate_proposals_compute_all_anchors", "generate_proposals.cl" },
    { "harris_score_3x3", "harris_corners.cl" },
    { "harris_score_5x5", "harris_corners.cl" },
    { "harris_score_7x7", "harris_corners.cl" },
    { "hist_border_kernel", "histogram.cl" },
    { "hist_border_kernel_fixed", "histogram.cl" },
    { "hist_local_kernel", "histogram.cl" },
    { "hist_local_kernel_fixed", "histogram.cl" },
    { "hog_block_normalization", "hog.cl" },
    { "hog_detector", "hog.cl" },
    { "hog_orientation_binning", "hog.cl" },
    { "hysteresis", "canny.cl" },
    { "im2col1x1_stridex1_nchw", "im2col.cl" },
    { "im2col3x3_nchw", "im2col.cl" },
    { "im2col5x5_nchw", "im2col.cl" },
    { "im2col11x11_padx0_pady0_nchw", "im2col.cl" },
    { "im2col_generic_nchw", "im2col.cl" },
    { "im2col_generic_padx0_pady0_nchw", "im2col.cl" },
    { "im2col3x3_nhwc", "im2col.cl" },
    { "im2col9x9_nhwc", "im2col.cl" },
    { "im2col_generic_nhwc", "im2col.cl" },
    { "init_level", "optical_flow_pyramid_lk.cl" },
    { "init_level_max", "optical_flow_pyramid_lk.cl" },
    { "init_level_max_initial_estimate", "optical_flow_pyramid_lk.cl" },
    { "integral_horizontal", "integral_image.cl" },
    { "integral_vertical", "integral_image.cl" },
    { "IYUV_to_NV12_bt709", "color_convert.cl" },
    { "IYUV_to_RGB888_bt709", "color_convert.cl" },
    { "IYUV_to_RGBA8888_bt709", "color_convert.cl" },
    { "IYUV_to_YUV444_bt709", "color_convert.cl" },
    { "l2_normalize_x", "l2_normalize.cl" },
    { "l2_normalize_y", "l2_normalize.cl" },
    { "l2_normalize_z", "l2_normalize.cl" },
    { "lktracker_stage0", "optical_flow_pyramid_lk.cl" },
    { "lktracker_stage1", "optical_flow_pyramid_lk.cl" },
    { "magnitude_phase", "magnitude_phase.cl" },
    { "mean_stddev_accumulate", "mean_stddev.cl" },
    { "memset", "memset.cl" },
    { "minmax", "minmaxloc.cl" },
    { "minmax_border", "minmaxloc.cl" },
    { "minmax_layer", "minmax_layer.cl" },
    { "minmaxloc", "minmaxloc.cl" },
    { "non_linear_filter_box3x3", "non_linear_filter3x3.cl" },
    { "non_linear_filter_cross3x3", "non_linear_filter3x3.cl" },
    { "non_linear_filter_disk3x3", "non_linear_filter3x3.cl" },
    { "non_linear_filter_box5x5", "non_linear_filter5x5.cl" },
    { "non_linear_filter_cross5x5", "non_linear_filter5x5.cl" },
    { "non_linear_filter_disk5x5", "non_linear_filter5x5.cl" },
    { "non_max_suppression", "nonmax.cl" },
    { "normalization_layer_cross_map", "normalization_layer.cl" },
    { "normalization_layer_in_map_nchw", "normalization_layer.cl" },
    { "normalization_layer_in_map_nhwc", "normalization_layer.cl" },
    { "normalize_planar_yuv_layer_nchw", "normalize_planar_yuv_layer.cl" },
    { "normalize_planar_yuv_layer_nhwc", "normalize_planar_yuv_layer.cl" },
    { "normalize_planar_yuv_layer_q8_nchw", "normalize_planar_yuv_layer_quantized.cl" },
    { "normalize_planar_yuv_layer_q8_nhwc", "normalize_planar_yuv_layer_quantized.cl" },
    { "NV12_to_IYUV_bt709", "color_convert.cl" },
    { "NV12_to_RGB888_bt709", "color_convert.cl" },
    { "NV12_to_RGBA8888_bt709", "color_convert.cl" },
    { "NV12_to_YUV444_bt709", "color_convert.cl" },
    { "NV21_to_IYUV_bt709", "color_convert.cl" },
    { "NV21_to_RGB888_bt709", "color_convert.cl" },
    { "NV21_to_RGBA8888_bt709", "color_convert.cl" },
    { "NV21_to_YUV444_bt709", "color_convert.cl" },
    { "output_stage_quantized", "direct_convolution_1x1_3x3_5x5_quantized.cl" },
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
    { "pooling_layer_MxN_quantized_nhwc", "pooling_layer_quantized.cl" },
    { "pooling_layer_MxN_quantized_nchw", "pooling_layer_quantized.cl" },
    { "prior_box_layer_nchw", "prior_box_layer.cl" },
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
    { "RGB888_to_IYUV_bt709", "color_convert.cl" },
    { "RGB888_to_NV12_bt709", "color_convert.cl" },
    { "RGB888_to_RGBA8888_bt709", "color_convert.cl" },
    { "RGB888_to_U8_bt709", "color_convert.cl" },
    { "RGB888_to_YUV444_bt709", "color_convert.cl" },
    { "RGBA8888_to_IYUV_bt709", "color_convert.cl" },
    { "RGBA8888_to_NV12_bt709", "color_convert.cl" },
    { "RGBA8888_to_RGB888_bt709", "color_convert.cl" },
    { "RGBA8888_to_YUV444_bt709", "color_convert.cl" },
    { "roi_align_layer", "roi_align_layer.cl" },
    { "roi_pooling_layer", "roi_pooling_layer.cl" },
    { "scale_nearest_neighbour_nchw", "scale.cl" },
    { "scale_nearest_neighbour_nhwc", "scale.cl" },
    { "scale_bilinear_nchw", "scale.cl" },
    { "scale_bilinear_nhwc", "scale.cl" },
    { "scale_bilinear_quantized_nchw", "scale_quantized.cl" },
    { "scale_bilinear_quantized_nhwc", "scale_quantized.cl" },
    { "scharr3x3", "scharr_filter.cl" },
    { "select_same_rank", "select.cl" },
    { "select_different_rank_2", "select.cl" },
    { "select_different_rank_n", "select.cl" },
    { "sobel3x3", "sobel_filter.cl" },
    { "sobel_separable5x1", "sobel_filter.cl" },
    { "sobel_separable1x5", "sobel_filter.cl" },
    { "sobel_separable7x1", "sobel_filter.cl" },
    { "sobel_separable1x7", "sobel_filter.cl" },
    { "softmax_layer_norm", "softmax_layer.cl" },
    { "softmax_layer_norm_quantized", "softmax_layer_quantized.cl" },
    { "softmax_layer_max_shift_exp_sum_quantized_serial", "softmax_layer_quantized.cl" },
    { "softmax_layer_max_shift_exp_sum_quantized_parallel", "softmax_layer_quantized.cl" },
    { "softmax_layer_max_shift_exp_sum_serial", "softmax_layer.cl" },
    { "space_to_batch_nchw", "space_to_batch.cl" },
    { "space_to_batch_static_nchw", "space_to_batch.cl" },
    { "space_to_batch_nhwc", "space_to_batch.cl" },
    { "space_to_batch_static_nhwc", "space_to_batch.cl" },
    { "softmax_layer_max_shift_exp_sum_parallel", "softmax_layer.cl" },
    { "stack_layer", "stack_layer.cl" },
    { "strided_slice", "slice_ops.cl" },
    { "suppress_non_maximum", "canny.cl" },
    { "tablelookup_U8", "tablelookup.cl" },
    { "tablelookup_S16", "tablelookup.cl" },
    { "threshold_binary", "threshold.cl" },
    { "threshold_range", "threshold.cl" },
    { "tile", "tile.cl" },
    { "transpose", "transpose.cl" },
    { "UYVY422_to_IYUV_bt709", "color_convert.cl" },
    { "UYVY422_to_NV12_bt709", "color_convert.cl" },
    { "UYVY422_to_RGB888_bt709", "color_convert.cl" },
    { "UYVY422_to_RGBA8888_bt709", "color_convert.cl" },
    { "upsample_layer_nchw", "upsample_layer.cl" },
    { "upsample_layer_nhwc", "upsample_layer.cl" },
    { "warp_affine_nearest_neighbour", "warp_affine.cl" },
    { "warp_affine_bilinear", "warp_affine.cl" },
    { "warp_perspective_nearest_neighbour", "warp_perspective.cl" },
    { "warp_perspective_bilinear", "warp_perspective.cl" },
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
    { "YUYV422_to_IYUV_bt709", "color_convert.cl" },
    { "YUYV422_to_NV12_bt709", "color_convert.cl" },
    { "YUYV422_to_RGB888_bt709", "color_convert.cl" },
    { "YUYV422_to_RGBA8888_bt709", "color_convert.cl" },
};

const std::map<std::string, std::string> CLKernelLibrary::_program_source_map =
{
#ifdef EMBEDDED_KERNELS
    {
        "absdiff.cl",
#include "./cl_kernels/absdiff.clembed"
    },
    {
        "accumulate.cl",
#include "./cl_kernels/accumulate.clembed"
    },
    {
        "activation_layer.cl",
#include "./cl_kernels/activation_layer.clembed"
    },
    {
        "activation_layer_qa8.cl",
#include "./cl_kernels/activation_layer_qa8.clembed"
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
        "canny.cl",
#include "./cl_kernels/canny.clembed"
    },
    {
        "channel_combine.cl",
#include "./cl_kernels/channel_combine.clembed"
    },
    {
        "channel_extract.cl",
#include "./cl_kernels/channel_extract.clembed"
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
        "color_convert.cl",
#include "./cl_kernels/color_convert.clembed"
    },
    {
        "convert_fc_weights.cl",
#include "./cl_kernels/convert_fc_weights.clembed"
    },
    {
        "convolution3x3.cl",
#include "./cl_kernels/convolution3x3.clembed"
    },
    {
        "convolution5x5.cl",
#include "./cl_kernels/convolution5x5.clembed"
    },
    {
        "convolution7x7.cl",
#include "./cl_kernels/convolution7x7.clembed"
    },
    {
        "convolution9x9.cl",
#include "./cl_kernels/convolution9x9.clembed"
    },
    {
        "convolution_layer.cl",
#include "./cl_kernels/convolution_layer.clembed"
    },
    {
        "convolution_rectangle.cl",
#include "./cl_kernels/convolution_rectangle.clembed"
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
        "derivative.cl",
#include "./cl_kernels/derivative.clembed"
    },
    {
        "dilate.cl",
#include "./cl_kernels/dilate.clembed"
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
        "direct_convolution_1x1_3x3_5x5_quantized.cl",
#include "./cl_kernels/direct_convolution_1x1_3x3_5x5_quantized.clembed"
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
        "erode.cl",
#include "./cl_kernels/erode.clembed"
    },
    {
        "fast_corners.cl",
#include "./cl_kernels/fast_corners.clembed"
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
        "flatten.cl",
#include "./cl_kernels/flatten.clembed"
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
        "gaussian_pyramid.cl",
#include "./cl_kernels/gaussian_pyramid.clembed"
    },
    {
        "gemm.cl",
#include "./cl_kernels/gemm.clembed"
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
        "harris_corners.cl",
#include "./cl_kernels/harris_corners.clembed"
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
        "histogram.cl",
#include "./cl_kernels/histogram.clembed"
    },
    {
        "hog.cl",
#include "./cl_kernels/hog.clembed"
    },
    {
        "im2col.cl",
#include "./cl_kernels/im2col.clembed"
    },
    {
        "integral_image.cl",
#include "./cl_kernels/integral_image.clembed"
    },
    {
        "l2_normalize.cl",
#include "./cl_kernels/l2_normalize.clembed"
    },
    {
        "magnitude_phase.cl",
#include "./cl_kernels/magnitude_phase.clembed"
    },
    {
        "mean_stddev.cl",
#include "./cl_kernels/mean_stddev.clembed"
    },
    {
        "memset.cl",
#include "./cl_kernels/memset.clembed"
    },
    {
        "minmaxloc.cl",
#include "./cl_kernels/minmaxloc.clembed"
    },
    {
        "minmax_layer.cl",
#include "./cl_kernels/minmax_layer.clembed"
    },
    {
        "non_linear_filter3x3.cl",
#include "./cl_kernels/non_linear_filter3x3.clembed"
    },
    {
        "non_linear_filter5x5.cl",
#include "./cl_kernels/non_linear_filter5x5.clembed"
    },
    {
        "non_linear_filter_helpers.h",
#include "./cl_kernels/non_linear_filter_helpers.hembed"
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
        "optical_flow_pyramid_lk.cl",
#include "./cl_kernels/optical_flow_pyramid_lk.clembed"
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
        "scharr_filter.cl",
#include "./cl_kernels/scharr_filter.clembed"
    },
    {
        "select.cl",
#include "./cl_kernels/select.clembed"
    },
    {
        "sobel_filter.cl",
#include "./cl_kernels/sobel_filter.clembed"
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
        "stack_layer.cl",
#include "./cl_kernels/stack_layer.clembed"
    },
    {
        "tablelookup.cl",
#include "./cl_kernels/tablelookup.clembed"
    },
    {
        "threshold.cl",
#include "./cl_kernels/threshold.clembed"
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
        "warp_affine.cl",
#include "./cl_kernels/warp_affine.clembed"
    },
    {
        "warp_helpers.h",
#include "./cl_kernels/warp_helpers.hembed"
    },
    {
        "warp_perspective.cl",
#include "./cl_kernels/warp_perspective.clembed"
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
    : _context(), _device(), _kernel_path("."), _programs_map(), _built_programs_map()
{
    opencl_is_available(); // Make sure the OpenCL symbols are initialised *before* the CLKernelLibrary is built
}

CLKernelLibrary &CLKernelLibrary::get()
{
    static CLKernelLibrary _kernel_library;
    return _kernel_library;
}

Kernel CLKernelLibrary::create_kernel(const std::string &kernel_name, const StringSet &build_options_set) const
{
    // Find which program contains the kernel
    auto kernel_program_it = _kernel_program_map.find(kernel_name);

    if(_kernel_program_map.end() == kernel_program_it)
    {
        ARM_COMPUTE_ERROR("Kernel %s not found in the CLKernelLibrary", kernel_name.c_str());
    }
    std::string concat_str;

#if defined(ARM_COMPUTE_DEBUG_ENABLED)
    // Enable debug properties in CL kernels
    concat_str += " -DARM_COMPUTE_DEBUG_ENABLED";
#endif // defined(ARM_COMPUTE_DEBUG_ENABLED)

    GPUTarget gpu_arch = get_arch_from_target(get_target_from_device(_device));
    concat_str += " -DGPU_ARCH=" + support::cpp11::to_string(
                      static_cast<std::underlying_type<GPUTarget>::type>(gpu_arch));
    if(fp16_supported())
    {
        concat_str += " -DARM_COMPUTE_OPENCL_FP16_ENABLED=1 ";
    }

    if(dot8_supported(_device))
    {
        concat_str += " -DARM_COMPUTE_OPENCL_DOT8_ENABLED=1 ";
    }

    if(dot8_acc_supported(_device))
    {
        concat_str += " -DARM_COMPUTE_OPENCL_DOT8_ACC_ENABLED=1 ";
    }

    if(get_cl_version(_device) == CLVersion::CL20)
    {
        concat_str += " -cl-std=CL2.0 ";
    }
    else if(arm_non_uniform_workgroup_supported(_device))
    {
        concat_str += " -cl-arm-non-uniform-work-group-size ";
    }
    else
    {
        ARM_COMPUTE_ERROR("Non uniform workgroup size is not supported!!");
    }

    // Check if the program has been built before with same build options.
    const std::string program_name  = kernel_program_it->second;
    const std::string build_options = stringify_set(build_options_set) + concat_str;

    const std::string built_program_name = program_name + "_" + build_options;
    auto              built_program_it   = _built_programs_map.find(built_program_name);

    cl::Program cl_program;

    if(_built_programs_map.end() != built_program_it)
    {
        // If program has been built, retrieve to create kernel from it
        cl_program = built_program_it->second;
    }
    else
    {
        // Get program
        Program program = load_program(program_name);

        // Build program
        cl_program = program.build(build_options);

        // Add built program to internal map
        _built_programs_map.emplace(built_program_name, cl_program);
    }

    // Create and return kernel
    return Kernel(kernel_name, cl_program);
}

void CLKernelLibrary::add_built_program(const std::string &built_program_name, const cl::Program &program)
{
    _built_programs_map.emplace(built_program_name, program);
}

bool CLKernelLibrary::fp16_supported() const
{
    return ::fp16_supported(_device);
}

bool CLKernelLibrary::int64_base_atomics_supported() const
{
    return device_supports_extension(_device, "cl_khr_int64_base_atomics");
}

const Program &CLKernelLibrary::load_program(const std::string &program_name) const
{
    const auto program_it = _programs_map.find(program_name);

    if(program_it != _programs_map.end())
    {
        return program_it->second;
    }

    Program program;

#ifdef EMBEDDED_KERNELS
    const auto program_source_it = _program_source_map.find(program_name);

    if(_program_source_map.end() == program_source_it)
    {
        ARM_COMPUTE_ERROR("Embedded program for %s does not exist.", program_name.c_str());
    }

    program = Program(_context, program_name, program_source_it->second);
#else  /* EMBEDDED_KERNELS */
    // Check for binary
    std::string source_name = _kernel_path + program_name;
    std::string binary_name = source_name + "bin";

    if(std::ifstream(binary_name).is_open())
    {
        const std::string program_binary = read_file(binary_name, true);
        program                          = Program(_context, _device, program_name, std::vector<unsigned char>(program_binary.begin(), program_binary.end()));
    }
    else if(std::ifstream(source_name).is_open())
    {
        program = Program(_context, program_name, read_file(source_name, false));
    }
    else
    {
        ARM_COMPUTE_ERROR("Kernel file %s does not exist.", source_name.c_str());
    }
#endif /* EMBEDDED_KERNELS */

    // Insert program to program map
    const auto new_program = _programs_map.emplace(program_name, std::move(program));

    return new_program.first->second;
}

std::string CLKernelLibrary::stringify_set(const StringSet &s) const
{
    std::string concat_set;

#ifndef EMBEDDED_KERNELS
    concat_set += "-I" + _kernel_path + " ";
#endif /* EMBEDDED_KERNELS */

    // Concatenate set
    for(const auto &el : s)
    {
        concat_set += " " + el;
    }

    return concat_set;
}

std::string CLKernelLibrary::get_program_source(const std::string &program_name)
{
    const auto program_source_it = _program_source_map.find(program_name);

    if(program_source_it == _program_source_map.end())
    {
        ARM_COMPUTE_ERROR("Embedded program for %s does not exist.", program_name.c_str());
    }

    return program_source_it->second;
}

size_t CLKernelLibrary::max_local_workgroup_size(const cl::Kernel &kernel) const
{
    size_t result;

    size_t err = kernel.getWorkGroupInfo(_device, CL_KERNEL_WORK_GROUP_SIZE, &result);
    ARM_COMPUTE_ERROR_ON_MSG(err != 0, "clGetKernelWorkGroupInfo failed to return the maximum workgroup size for the kernel");
    ARM_COMPUTE_UNUSED(err);

    return result;
}

cl::NDRange CLKernelLibrary::default_ndrange() const
{
    GPUTarget   _target = get_target_from_device(_device);
    cl::NDRange default_range;

    switch(_target)
    {
        case GPUTarget::MIDGARD:
        case GPUTarget::T600:
        case GPUTarget::T700:
        case GPUTarget::T800:
            default_range = cl::NDRange(128u, 1);
            break;
        default:
            default_range = cl::NullRange;
    }

    return default_range;
}

std::string CLKernelLibrary::get_device_version()
{
    return _device.getInfo<CL_DEVICE_VERSION>();
}
