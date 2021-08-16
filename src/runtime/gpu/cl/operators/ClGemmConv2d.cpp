/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "src/runtime/gpu/cl/operators/ClGemmConv2d.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "src/core/gpu/cl/kernels/ClActivationKernel.h"
#include "src/core/gpu/cl/kernels/ClCol2ImKernel.h"
#include "src/core/gpu/cl/kernels/ClIm2ColKernel.h"
#include "src/core/gpu/cl/kernels/ClWeightsReshapeKernel.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/runtime/gpu/cl/operators/ClGemm.h"
#include "src/runtime/gpu/cl/operators/ClGemmLowpMatrixMultiplyCore.h"
#include "src/runtime/gpu/cl/utils/ClAuxTensorHandler.h"
#include "support/Cast.h"

namespace arm_compute
{
using namespace experimental;
using namespace misc::shape_calculator;
using namespace utils::cast;
namespace opencl
{
ClGemmConv2d::ClGemmConv2d()
    : _weights_reshape_kernel(nullptr), _im2col_kernel(nullptr), _mm_gemm(nullptr), _mm_gemmlowp(nullptr), _col2im_kernel(nullptr), _activation_kernel(nullptr), _im2col_output(), _weights_reshaped(),
      _gemm_output(), _skip_im2col(false), _skip_col2im(false), _is_quantized(false), _fuse_activation(true), _append_bias(false), _is_prepared(false), _aux_mem(AuxTensorIdx::Count)
{
}
ClGemmConv2d::~ClGemmConv2d() = default;

void ClGemmConv2d::configure_mm(const ClCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *weights, ITensorInfo *biases, ITensorInfo *dst,
                                const GEMMLowpOutputStageInfo &gemmlowp_output_stage,
                                int gemm_3d_depth, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, weights);
    ARM_COMPUTE_ERROR_THROW_ON(validate_mm(src, weights, biases, dst, gemmlowp_output_stage, gemm_3d_depth, _skip_im2col, act_info));

    const GEMMInfo &gemm_info = GEMMInfo(false,                 // is_a_reshaped
                                         false,                 // is_b_reshaped
                                         true,                  // reshape_b_only_on_first_run
                                         gemm_3d_depth,         // depth_output_gemm3d
                                         _skip_im2col,          // reinterpret_input_as_3d
                                         false,                 // retain_internal_weights
                                         gemmlowp_output_stage, // gemmlowp_output_stage
                                         false,                 // fast_math
                                         false,                 // fp_mixed_precision
                                         true,                  // broadcast_bias
                                         act_info);             // activation_info

    TensorInfo tmp_src{ *src };
    if(_is_quantized)
    {
        // Since we need negative offsets for computing convolution, we need to change QuantizationInfo()
        // Extract and negate input and weights offset
        const QuantizationInfo input_quantization_info   = src->quantization_info();
        const QuantizationInfo weights_quantization_info = weights->quantization_info();

        tmp_src.set_quantization_info(QuantizationInfo(input_quantization_info.uniform().scale, -input_quantization_info.uniform().offset));
        weights->set_quantization_info(QuantizationInfo(weights_quantization_info.uniform().scale, -weights_quantization_info.uniform().offset));

        _mm_gemmlowp = std::make_unique<ClGemmLowpMatrixMultiplyCore>();
        _mm_gemmlowp->configure(compile_context, &tmp_src, weights, biases, dst, gemm_info);

        // Revert back QuantizatioInfo as weights could be used in other convolution layers
        weights->set_quantization_info(weights_quantization_info);

        auto mm_mem_req = _mm_gemmlowp->workspace();
        for(unsigned int cont = 0; cont < mm_mem_req.size(); ++cont)
        {
            _aux_mem[cont] = mm_mem_req[cont];
        }
    }
    else
    {
        // Configure matrix multiply function
        _mm_gemm = std::make_unique<ClGemm>();
        _mm_gemm->configure(compile_context, &tmp_src, weights, biases, dst, 1.0f, 1.0f, gemm_info);
        auto mm_mem_req = _mm_gemm->workspace();
        for(unsigned int cont = 0; cont < mm_mem_req.size(); ++cont)
        {
            _aux_mem[cont] = mm_mem_req[cont];
        }
    }
}

Status ClGemmConv2d::validate_mm(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst,
                                 const GEMMLowpOutputStageInfo &gemmlowp_output_stage, int gemm_3d_depth, bool skip_im2col, const ActivationLayerInfo &act_info)
{
    const bool is_quantized = is_data_type_quantized_asymmetric(src->data_type());

    const GEMMInfo &gemm_info = GEMMInfo(false,                 // is_a_reshaped
                                         false,                 // is_b_reshaped
                                         true,                  // reshape_b_only_on_first_run
                                         gemm_3d_depth,         // depth_output_gemm3d
                                         skip_im2col,           // reinterpret_input_as_3d
                                         false,                 // retain_internal_weights
                                         gemmlowp_output_stage, // gemmlowp_output_stage
                                         false,                 // fast_math
                                         false,                 // fp_mixed_precision
                                         true,                  // broadcast_bias
                                         act_info);             // activation_info

    if(is_quantized)
    {
        // Since we need negative offsets for computing convolution, we need to change QuantizationInfo()
        // Extract and negate input and weights offset
        const QuantizationInfo input_quantization_info   = src->quantization_info();
        const QuantizationInfo weights_quantization_info = weights->quantization_info();

        std::unique_ptr<ITensorInfo> src_qa     = src->clone();
        std::unique_ptr<ITensorInfo> weights_qa = weights->clone();
        src_qa->set_quantization_info(QuantizationInfo(input_quantization_info.uniform().scale, -input_quantization_info.uniform().offset));
        weights_qa->set_quantization_info(QuantizationInfo(weights_quantization_info.uniform().scale, -weights_quantization_info.uniform().offset));

        // Perform validation step on GEMMLowp
        return ClGemmLowpMatrixMultiplyCore::validate(src_qa.get(), weights_qa.get(), biases, dst, gemm_info);
    }
    else
    {
        // Perform validation step on Matrix multiply function
        return ClGemm::validate(src, weights, biases, dst, 1.0f, 1.0f, gemm_info);
    }
}

void ClGemmConv2d::configure(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *weights, ITensorInfo *biases, ITensorInfo *dst,
                             const Conv2dInfo &conv2d_info, const WeightsInfo &weights_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, weights, dst);

    ARM_COMPUTE_ERROR_THROW_ON(ClGemmConv2d::validate(src, weights, biases, dst,
                                                      conv2d_info,
                                                      weights_info));

    const DataType   data_type   = src->data_type();
    const DataLayout data_layout = src->data_layout();
    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        idx_kernels = get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES);

    const unsigned int kernel_width  = weights->dimension(idx_width);
    const unsigned int kernel_height = weights->dimension(idx_height);
    const unsigned int num_kernels   = weights->dimension(idx_kernels);

    const UniformQuantizationInfo iq_info = src->quantization_info().uniform();
    const UniformQuantizationInfo oq_info = dst->quantization_info().uniform();

    _is_prepared  = weights_info.retain_internal_weights();
    _is_quantized = is_data_type_quantized_asymmetric(src->data_type());
    _skip_im2col  = (data_layout == DataLayout::NHWC && kernel_width == 1 && kernel_height == 1 && conv2d_info.conv_info.stride().first == 1 && conv2d_info.conv_info.stride().second == 1);
    _skip_col2im  = data_layout == DataLayout::NHWC;

    // Only for quantize there are few cases where we cannot fuse the activation function in GEMM
    _fuse_activation = true;

    const ITensorInfo *gemm_input_to_use  = src;
    ITensorInfo       *gemm_output_to_use = dst;

    // Get parameters from conv_info
    unsigned int stride_x = 0;
    unsigned int stride_y = 0;
    std::tie(stride_x, stride_y) = conv2d_info.conv_info.stride();

    // Get convolved dimensions
    unsigned int conv_w = 0;
    unsigned int conv_h = 0;
    std::tie(conv_w, conv_h) = scaled_dimensions(src->dimension(idx_width),
                                                 src->dimension(idx_height),
                                                 kernel_width,
                                                 kernel_height,
                                                 conv2d_info.conv_info,
                                                 conv2d_info.dilation);

    unsigned int mat_weights_cols = num_kernels / conv2d_info.num_groups;

    ITensorInfo *biases_to_use = biases;
    _append_bias               = false;

    _weights_reshape_kernel = std::make_unique<kernels::ClWeightsReshapeKernel>();
    if(conv2d_info.num_groups != 1 && biases != nullptr)
    {
        // num_groups != 1 can only be for NCHW
        // Since it is missing an utility function to reshape the biases, we append the biases into the weights tensor
        biases_to_use = nullptr;
        _append_bias  = true;
        _weights_reshape_kernel->configure(compile_context, weights, biases, &_weights_reshaped, conv2d_info.num_groups);
    }
    else
    {
        _weights_reshape_kernel->configure(compile_context, weights, nullptr, &_weights_reshaped, conv2d_info.num_groups);
    }

    // Create tensor to store im2col reshaped inputs
    if(!_skip_im2col)
    {
        // Configure and tune im2col. im2col output shape is auto-initialized
        _im2col_kernel = std::make_unique<opencl::kernels::ClIm2ColKernel>();

        // Set the GPU target for im2col
        _im2col_kernel->set_target(CLScheduler::get().target());
        _im2col_kernel->configure(compile_context, src, &_im2col_output, Size2D(kernel_width, kernel_height), conv2d_info.conv_info, _append_bias, conv2d_info.dilation, conv2d_info.num_groups);

        // Set quantization info
        _im2col_output.set_quantization_info(src->quantization_info());
        CLScheduler::get().tune_kernel_static(*_im2col_kernel);

        // Update GEMM input
        gemm_input_to_use = &_im2col_output;
    }

    // Create GEMM output tensor
    if(!_skip_col2im)
    {
        TensorShape shape_gemm;

        // If we cannot skip col2im it means we run im2col as well
        shape_gemm = _im2col_output.tensor_shape();
        shape_gemm.set(0, mat_weights_cols);
        shape_gemm.set(1, conv_w * conv_h);

        _gemm_output = TensorInfo(shape_gemm, 1, data_type);
        _gemm_output.set_quantization_info(dst->quantization_info()).set_data_layout(src->data_layout());

        // Update GEMM output
        gemm_output_to_use = &_gemm_output;
    }

    GEMMLowpOutputStageInfo gemmlowp_output_stage;
    gemmlowp_output_stage.type            = GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
    gemmlowp_output_stage.gemmlowp_offset = 0;

    // Configure output stage for quantized case
    if(_is_quantized)
    {
        const auto         output_quant_info        = (dst->total_size() == 0) ? iq_info : oq_info;
        const bool         is_quantized_per_channel = is_data_type_quantized_per_channel(weights->data_type());
        const unsigned int num_filters              = (is_quantized_per_channel) ? num_kernels : 1;

        gemmlowp_output_stage.is_quantized_per_channel = is_quantized_per_channel;

        gemmlowp_output_stage.gemmlowp_multipliers.resize(num_filters);
        gemmlowp_output_stage.gemmlowp_shifts.resize(num_filters);
        quantization::compute_quantized_multipliers_and_shifts(src, weights, dst,
                                                               gemmlowp_output_stage.gemmlowp_multipliers.data(),
                                                               gemmlowp_output_stage.gemmlowp_shifts.data());
        gemmlowp_output_stage.gemmlowp_multiplier = gemmlowp_output_stage.gemmlowp_multipliers[0];
        gemmlowp_output_stage.gemmlowp_shift      = gemmlowp_output_stage.gemmlowp_shifts[0];

        PixelValue min_val{};
        PixelValue max_val{};
        std::tie(min_val, max_val) = get_min_max(dst->data_type());

        auto min_activation = min_val.get<int32_t>();
        auto max_activation = max_val.get<int32_t>();

        const std::set<ActivationLayerInfo::ActivationFunction> supported_acts = { ActivationLayerInfo::ActivationFunction::RELU,
                                                                                   ActivationLayerInfo::ActivationFunction::BOUNDED_RELU,
                                                                                   ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU
                                                                                 };

        if(conv2d_info.act_info.enabled())
        {
            if(supported_acts.count(conv2d_info.act_info.activation()) != 0)
            {
                std::tie(min_activation, max_activation) = get_quantized_activation_min_max(conv2d_info.act_info, data_type, output_quant_info);
            }
            else
            {
                _fuse_activation = false;
            }
        }

        // Set the GEMMLowp output stage info
        gemmlowp_output_stage.gemmlowp_offset    = output_quant_info.offset;
        gemmlowp_output_stage.gemmlowp_min_bound = min_activation;
        gemmlowp_output_stage.gemmlowp_max_bound = max_activation;
    }

    // Configure and tune GEMM
    // In case of NHWC, we need to run GEMM3D (gemm_3d_depth != 0) in order to avoid reshaping the output matrix
    const unsigned int gemm_3d_depth = (data_layout == DataLayout::NHWC) ? conv_h : 0;

    configure_mm(compile_context, gemm_input_to_use, &_weights_reshaped, biases_to_use, gemm_output_to_use, gemmlowp_output_stage, gemm_3d_depth, conv2d_info.act_info);

    if(!_skip_col2im)
    {
        // Set the GPU target for col2im
        _col2im_kernel = std::make_unique<opencl::kernels::ClCol2ImKernel>();
        _col2im_kernel->set_target(CLScheduler::get().target());
        // Configure and tune Col2Im
        _col2im_kernel->configure(compile_context, gemm_output_to_use, dst, Size2D(conv_w, conv_h), conv2d_info.num_groups);
        CLScheduler::get().tune_kernel_static(*_col2im_kernel.get());
    }

    ARM_COMPUTE_ERROR_ON_MSG((dst->dimension(idx_width) != conv_w) || (dst->dimension(idx_height) != conv_h),
                             "Output shape does not match the expected one");

    if(!_fuse_activation)
    {
        _activation_kernel = std::make_unique<opencl::kernels::ClActivationKernel>();
        _activation_kernel->configure(compile_context, dst, nullptr, conv2d_info.act_info);
    }

    _aux_mem[Im2ColOutput]    = MemoryInfo(offset_int_vec(Im2ColOutput), MemoryLifetime::Temporary, _im2col_output.total_size());
    _aux_mem[WeightsReshaped] = MemoryInfo(offset_int_vec(WeightsReshaped), MemoryLifetime::Persistent, _weights_reshaped.total_size());
    _aux_mem[GemmOutput]      = MemoryInfo(offset_int_vec(GemmOutput), MemoryLifetime::Temporary, _gemm_output.total_size());
}

Status ClGemmConv2d::validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst, const Conv2dInfo &conv2d_info,
                              const WeightsInfo &weights_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, weights, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights_info.are_reshaped(), "Weights already reshaped are not supported!");
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);
    const bool is_quantized_per_channel = is_data_type_quantized_per_channel(weights->data_type());

    if(!is_quantized_per_channel)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, weights);
    }
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(src, weights);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((conv2d_info.num_groups != 1) && (src->data_layout() != DataLayout::NCHW), "Grouping (num_groups != 1) with NHWC data layout is not supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((conv2d_info.num_groups != 1) && (src->data_type() == DataType::QASYMM8), "Grouping (num_groups != 1) is not supported with QASYMM8");
    ARM_COMPUTE_RETURN_ERROR_ON(((src->dimension(2) / weights->dimension(2)) != conv2d_info.num_groups) && (src->data_layout() == DataLayout::NCHW));

    const DataLayout data_layout = src->data_layout();
    const DataType   data_type   = src->data_type();
    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        idx_channel = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);
    const int        idx_kernels = get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES);

    const unsigned int kernel_width  = weights->dimension(idx_width);
    const unsigned int kernel_height = weights->dimension(idx_height);
    const unsigned int num_kernels   = weights->dimension(idx_kernels);

    TensorInfo         im2col_reshaped_info{};
    TensorInfo         info_gemm{};
    TensorInfo         weights_reshaped_info{};
    const ITensorInfo *gemm_input_to_use  = src;
    const ITensorInfo *gemm_output_to_use = dst;
    const ITensorInfo *weights_to_use     = weights;
    const bool         is_quantized       = is_data_type_quantized_asymmetric(data_type);
    const bool         skip_im2col        = (data_layout == DataLayout::NHWC && kernel_width == 1 && kernel_height == 1 && conv2d_info.conv_info.stride().first == 1
                                             && conv2d_info.conv_info.stride().second == 1);
    const bool skip_col2im     = data_layout == DataLayout::NHWC;
    bool       fuse_activation = true;

    ARM_COMPUTE_RETURN_ERROR_ON((weights->dimension(idx_channel) * conv2d_info.num_groups) != src->dimension(idx_channel));
    ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 4);

    // Validate biases
    if(biases != nullptr)
    {
        if(is_quantized)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, DataType::S32);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, biases);
        }
        ARM_COMPUTE_RETURN_ERROR_ON(biases->dimension(0) != weights->dimension(idx_kernels));
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
    }

    if(conv2d_info.act_info.enabled())
    {
        ARM_COMPUTE_ERROR_ON(conv2d_info.act_info.b() > conv2d_info.act_info.a());
    }

    // Get convolved dimensions
    unsigned int conv_w = 0;
    unsigned int conv_h = 0;

    std::tie(conv_w, conv_h) = scaled_dimensions(src->dimension(idx_width),
                                                 src->dimension(idx_height),
                                                 kernel_width,
                                                 kernel_height,
                                                 conv2d_info.conv_info,
                                                 conv2d_info.dilation);

    unsigned int mat_weights_cols = num_kernels / conv2d_info.num_groups;

    const ITensorInfo *biases_to_use = biases;
    bool               append_bias   = false;

    if(conv2d_info.num_groups != 1 && biases != nullptr)
    {
        // num_groups != 1 can only be for NCHW
        // Since it is missing an utility function to reshape the biases, we append the biases into the weights tensor
        biases_to_use         = nullptr;
        append_bias           = true;
        weights_reshaped_info = TensorInfo(compute_weights_reshaped_shape(*weights, true, conv2d_info.num_groups), 1, data_type);
    }
    else
    {
        weights_reshaped_info = TensorInfo(compute_weights_reshaped_shape(*weights, false, conv2d_info.num_groups), 1, data_type);
    }

    weights_to_use = &weights_reshaped_info;

    if(!skip_im2col)
    {
        const Size2D kernel_dims(kernel_width, kernel_height);

        // Output tensor auto initialization if not yet initialized
        TensorShape expected_output_shape = compute_im2col_conv_shape(src, kernel_dims, conv2d_info.conv_info, append_bias, conv2d_info.dilation, conv2d_info.num_groups == 1, conv2d_info.num_groups);

        auto_init_if_empty(im2col_reshaped_info, src->clone()->set_tensor_shape(expected_output_shape));

        ARM_COMPUTE_RETURN_ON_ERROR(opencl::kernels::ClIm2ColKernel::validate(src, &im2col_reshaped_info, kernel_dims, conv2d_info.conv_info, append_bias, conv2d_info.dilation, conv2d_info.num_groups));
        gemm_input_to_use = &im2col_reshaped_info;
    }

    // Create GEMM output tensor
    if(!skip_col2im)
    {
        TensorShape shape_gemm;

        shape_gemm = gemm_input_to_use->tensor_shape();
        shape_gemm.set(0, mat_weights_cols);
        shape_gemm.set(1, conv_w * conv_h);

        info_gemm = TensorInfo(shape_gemm, 1, data_type);
        info_gemm.set_quantization_info(dst->quantization_info()).set_data_layout(src->data_layout());
        gemm_output_to_use = &info_gemm;
    }

    GEMMLowpOutputStageInfo gemmlowp_output_stage;
    gemmlowp_output_stage.type                     = GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
    gemmlowp_output_stage.gemmlowp_offset          = 0;
    gemmlowp_output_stage.is_quantized_per_channel = is_quantized_per_channel;

    if(is_quantized)
    {
        const UniformQuantizationInfo iq_info           = src->quantization_info().uniform();
        const UniformQuantizationInfo oq_info           = dst->quantization_info().uniform();
        const auto                    output_quant_info = (dst->total_size() == 0) ? iq_info : oq_info;
        const unsigned int            num_filters       = (is_quantized_per_channel) ? num_kernels : 1;

        gemmlowp_output_stage.gemmlowp_multipliers.resize(num_filters);
        gemmlowp_output_stage.gemmlowp_shifts.resize(num_filters);
        quantization::compute_quantized_multipliers_and_shifts(src, weights, dst,
                                                               gemmlowp_output_stage.gemmlowp_multipliers.data(),
                                                               gemmlowp_output_stage.gemmlowp_shifts.data());
        gemmlowp_output_stage.gemmlowp_multiplier = gemmlowp_output_stage.gemmlowp_multipliers[0];
        gemmlowp_output_stage.gemmlowp_shift      = gemmlowp_output_stage.gemmlowp_shifts[0];

        int min_activation = 0;
        int max_activation = 0;

        const std::set<ActivationLayerInfo::ActivationFunction> supported_acts = { ActivationLayerInfo::ActivationFunction::RELU,
                                                                                   ActivationLayerInfo::ActivationFunction::BOUNDED_RELU,
                                                                                   ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU
                                                                                 };

        if(conv2d_info.act_info.enabled())
        {
            if(supported_acts.count(conv2d_info.act_info.activation()) != 0)
            {
                std::tie(min_activation, max_activation) = get_quantized_activation_min_max(conv2d_info.act_info, data_type, output_quant_info);
            }
            else
            {
                fuse_activation = false;
            }
        }

        // Set the GEMMLowp output stage info
        gemmlowp_output_stage.gemmlowp_offset    = output_quant_info.offset;
        gemmlowp_output_stage.gemmlowp_min_bound = min_activation;
        gemmlowp_output_stage.gemmlowp_max_bound = max_activation;
    }

    // In case of NHWC, we need to run GEMM3D (gemm_3d_depth != 0) in order to avoid reshaping the output matrix
    const unsigned int gemm_3d_depth = (data_layout == DataLayout::NHWC) ? conv_h : 0;

    ARM_COMPUTE_RETURN_ON_ERROR(validate_mm(gemm_input_to_use, weights_to_use, biases_to_use, gemm_output_to_use, gemmlowp_output_stage, gemm_3d_depth, skip_im2col, conv2d_info.act_info));

    // Validate Col2Im
    if(!skip_col2im)
    {
        ARM_COMPUTE_RETURN_ON_ERROR(kernels::ClCol2ImKernel::validate(gemm_output_to_use, dst, Size2D(conv_w, conv_h), conv2d_info.num_groups));
    }

    //Validate Activation Layer
    if(!fuse_activation)
    {
        ARM_COMPUTE_RETURN_ON_ERROR(kernels::ClActivationKernel::validate(dst, nullptr, conv2d_info.act_info));
    }

    return Status{};
}

void ClGemmConv2d::run(ITensorPack &tensors)
{
    prepare(tensors);

    auto src                = tensors.get_const_tensor(ACL_SRC_0);
    auto biases             = tensors.get_const_tensor(ACL_SRC_2);
    auto dst                = tensors.get_tensor(ACL_DST);
    auto gemm_input_to_use  = src;
    auto gemm_output_to_use = dst;

    CLAuxTensorHandler im2col_output(offset_int_vec(Im2ColOutput), _im2col_output, tensors, false);
    CLAuxTensorHandler gemm_output(offset_int_vec(GemmOutput), _gemm_output, tensors, false);
    CLAuxTensorHandler weights_reshaped(offset_int_vec(WeightsReshaped), _weights_reshaped, tensors, false);

    // Run im2col
    if(!_skip_im2col)
    {
        ITensorPack pack =
        {
            { TensorType::ACL_SRC, src },
            { TensorType::ACL_DST, im2col_output.get() }
        };
        CLScheduler::get().enqueue_op(*_im2col_kernel, pack, false);
        gemm_input_to_use = im2col_output.get();
    }
    if(!_skip_col2im)
    {
        gemm_output_to_use = gemm_output.get();
    }
    ITensorPack pack_mm = tensors;
    pack_mm.add_const_tensor(TensorType::ACL_SRC_0, gemm_input_to_use);
    pack_mm.add_const_tensor(TensorType::ACL_SRC_1, weights_reshaped.get());
    if(!_append_bias)
    {
        pack_mm.add_const_tensor(TensorType::ACL_SRC_2, biases);
    }
    pack_mm.add_tensor(TensorType::ACL_DST, gemm_output_to_use);
    // Runs ClGemm or ClGemmLowpMatrixMultiplyCore functions
    if(_is_quantized)
    {
        // Run gemmlowp
        _mm_gemmlowp->run(pack_mm);
    }
    else
    {
        // Run gemm
        _mm_gemm->run(pack_mm);
    }

    // Reshape output matrix
    if(!_skip_col2im)
    {
        ITensorPack pack =
        {
            { TensorType::ACL_SRC, gemm_output_to_use },
            { TensorType::ACL_DST, dst }
        };
        CLScheduler::get().enqueue_op(*_col2im_kernel.get(), pack, false);
    }

    //Run Activation Layer if we cannot fuse in GEMM
    if(!_fuse_activation)
    {
        ITensorPack pack =
        {
            { TensorType::ACL_SRC, dst },
            { TensorType::ACL_DST, dst }
        };
        CLScheduler::get().enqueue_op(*_activation_kernel.get(), pack, false);
    }
}

void ClGemmConv2d::prepare(ITensorPack &tensors)
{
    if(!_is_prepared)
    {
        // Run weights reshaping and mark original weights tensor as unused
        ICLTensor         *weights_reshaped_p = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(offset_int_vec(WeightsReshaped)));
        CLAuxTensorHandler weights_reshaped(_weights_reshaped, *weights_reshaped_p);
        auto               weights = tensors.get_const_tensor(TensorType::ACL_SRC_1);
        ITensorPack        pack =
        {
            { TensorType::ACL_SRC, weights },
            { TensorType::ACL_DST, weights_reshaped.get() }
        };

        if(_append_bias)
        {
            const auto biases = tensors.get_const_tensor(TensorType::ACL_SRC_2);
            pack.add_const_tensor(TensorType::ACL_BIAS, biases);
        }
        CLScheduler::get().enqueue_op(*_weights_reshape_kernel.get(), pack, true);
        tensors.add_const_tensor(TensorType::ACL_SRC_1, weights_reshaped.get());

        // Prepare GEMM
        _is_quantized ? _mm_gemmlowp->prepare(tensors) : _mm_gemm->prepare(tensors);
        _is_prepared = true;
    }
}
experimental::MemoryRequirements ClGemmConv2d::workspace() const
{
    return _aux_mem;
}
} // namespace opencl
} // namespace arm_compute
