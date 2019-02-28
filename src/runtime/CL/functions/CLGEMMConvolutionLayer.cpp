/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLGEMMConvolutionLayer.h"

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include <cmath>
#include <memory>
#include <tuple>

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

CLConvolutionLayerReshapeWeights::CLConvolutionLayerReshapeWeights()
    : _weights_reshape_kernel()
{
}

void CLConvolutionLayerReshapeWeights::configure(const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, unsigned int num_groups)
{
    // Perform validation step
    ARM_COMPUTE_ERROR_ON_NULLPTR(weights, output);
    ARM_COMPUTE_ERROR_THROW_ON(CLConvolutionLayerReshapeWeights::validate(weights->info(),
                                                                          (biases != nullptr) ? biases->info() : nullptr,
                                                                          output->info(),
                                                                          num_groups));

    const bool       append_biases = (biases != nullptr) && !is_data_type_quantized_asymmetric(weights->info()->data_type());
    const ICLTensor *biases_to_use = (append_biases) ? biases : nullptr;

    _weights_reshape_kernel.configure(weights, biases_to_use, output, num_groups);

    output->info()->set_quantization_info(weights->info()->quantization_info());
}

Status CLConvolutionLayerReshapeWeights::validate(const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, unsigned int num_groups)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(weights);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 4);

    if(biases != nullptr)
    {
        const int idx_kernels = get_data_layout_dimension_index(weights->data_layout(), DataLayoutDimension::BATCHES);
        ARM_COMPUTE_RETURN_ERROR_ON(is_data_type_quantized_asymmetric(weights->data_type()));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(weights, biases);
        ARM_COMPUTE_RETURN_ERROR_ON(biases->dimension(0) != weights->dimension(idx_kernels));
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
    }

    if((output != nullptr) && (output->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(weights, output);

        CLWeightsReshapeKernel::validate(weights, biases, output, num_groups);
    }

    return Status{};
}

void CLConvolutionLayerReshapeWeights::run()
{
    CLScheduler::get().enqueue(_weights_reshape_kernel);
}

CLGEMMConvolutionLayer::CLGEMMConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(memory_manager), _reshape_weights(), _im2col_kernel(), _mm_gemm(memory_manager), _mm_gemmlowp(memory_manager), _col2im_kernel(), _activationlayer_function(), _add_bias_kernel(),
      _original_weights(nullptr), _im2col_output(), _weights_reshaped(), _gemm_output(), _data_layout(DataLayout::NCHW), _append_bias(false), _skip_im2col(false), _skip_col2im(false), _is_quantized(false),
      _is_activationlayer_enabled(false), _is_prepared(false), _run_addition(true)
{
}

void CLGEMMConvolutionLayer::configure_mm(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const GEMMLowpOutputStageInfo &gemmlowp_output_stage,
                                          int gemm_3d_depth)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights);
    ARM_COMPUTE_ERROR_THROW_ON(validate_mm(input->info(), weights->info(), biases != nullptr ? biases->info() : nullptr, output->info(), gemmlowp_output_stage, gemm_3d_depth, _skip_im2col,
                                           _run_addition));

    const GEMMInfo &gemm_info = GEMMInfo(false, false, true /* Reshape weights only for the first run */,
                                         gemm_3d_depth, _skip_im2col /* Reinterpret the input as 3D if im2col is skipped */,
                                         false, gemmlowp_output_stage);

    if(_is_quantized)
    {
        // Since we need negative offsets for computing convolution, we need to change QuantizationInfo()
        // Extract and negate input and weights offset
        const QuantizationInfo input_quantization_info   = input->info()->quantization_info();
        const QuantizationInfo weights_quantization_info = weights->info()->quantization_info();

        input->info()->set_quantization_info(QuantizationInfo(input_quantization_info.scale, -input_quantization_info.offset));
        weights->info()->set_quantization_info(QuantizationInfo(weights_quantization_info.scale, -weights_quantization_info.offset));

        _mm_gemmlowp.configure(input, weights, biases, output, gemm_info);

        // Revert back QuantizatioInfo as input and weights could be used in other convolution layers
        input->info()->set_quantization_info(input_quantization_info);
        weights->info()->set_quantization_info(weights_quantization_info);
    }
    else
    {
        // Bias does not need to be added in GEMM if im2col is being used or the Matrix Addition kernel needs to be run
        const bool skip_bias_in_gemm = _run_addition || !_skip_im2col;
        // Configure matrix multiply function
        _mm_gemm.configure(input, weights, (skip_bias_in_gemm) ? nullptr : biases, output, 1.0f, 1.0f, gemm_info);
    }
}

Status CLGEMMConvolutionLayer::validate_mm(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output,
                                           const GEMMLowpOutputStageInfo &gemmlowp_output_stage, int gemm_3d_depth, bool skip_im2col, bool run_addition)
{
    const bool is_quantized = is_data_type_quantized_asymmetric(input->data_type());

    const GEMMInfo &gemm_info = GEMMInfo(false, false, true /* Reshape weights only for the first run */,
                                         gemm_3d_depth, skip_im2col /* Reinterpret the input as 3D if im2col is skipped */,
                                         false, gemmlowp_output_stage);

    if(is_quantized)
    {
        // Since we need negative offsets for computing convolution, we need to change QuantizationInfo()
        // Extract and negate input and weights offset
        const QuantizationInfo input_quantization_info   = input->quantization_info();
        const QuantizationInfo weights_quantization_info = weights->quantization_info();

        std::unique_ptr<ITensorInfo> input_qa   = input->clone();
        std::unique_ptr<ITensorInfo> weights_qa = weights->clone();
        input_qa->set_quantization_info(QuantizationInfo(input_quantization_info.scale, -input_quantization_info.offset));
        weights_qa->set_quantization_info(QuantizationInfo(weights_quantization_info.scale, -weights_quantization_info.offset));

        // Perform validation step on GEMMLowp
        return CLGEMMLowpMatrixMultiplyCore::validate(input_qa.get(), weights_qa.get(), biases, output, gemm_info);
    }
    else
    {
        // Bias does not need to be added in GEMM if im2col is being used or the Matrix Addition kernel needs to be run
        const bool skip_bias_in_gemm = run_addition || !skip_im2col;
        // Perform validation step on Matrix multiply function
        return CLGEMM::validate(input, weights, (skip_bias_in_gemm) ? nullptr : biases, output, 1.0f, 1.0f, gemm_info);
    }
}

void CLGEMMConvolutionLayer::configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info,
                                       const Size2D &dilation, const ActivationLayerInfo &act_info, unsigned int num_groups)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);

    ARM_COMPUTE_ERROR_THROW_ON(CLGEMMConvolutionLayer::validate(input->info(),
                                                                weights->info(),
                                                                biases != nullptr ? biases->info() : nullptr,
                                                                output->info(),
                                                                conv_info,
                                                                weights_info,
                                                                dilation,
                                                                act_info,
                                                                num_groups));

    const DataType   data_type   = input->info()->data_type();
    const DataLayout data_layout = input->info()->data_layout();
    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        idx_kernels = get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES);

    const unsigned int kernel_width  = weights->info()->dimension(idx_width);
    const unsigned int kernel_height = weights->info()->dimension(idx_height);

    _is_prepared                = weights_info.retain_internal_weights();
    _original_weights           = weights;
    _is_quantized               = is_data_type_quantized_asymmetric(input->info()->data_type());
    _data_layout                = data_layout;
    _skip_im2col                = (data_layout == DataLayout::NHWC && kernel_width == 1 && kernel_height == 1 && conv_info.stride().first == 1 && conv_info.stride().second == 1);
    _skip_col2im                = data_layout == DataLayout::NHWC;
    _append_bias                = (biases != nullptr) && (!_is_quantized);
    _is_activationlayer_enabled = act_info.enabled();
    // In case of F16, fused bias will be used in GEMM
    _run_addition = (_skip_im2col) && (_append_bias) && (data_type != DataType::F16);

    // Set the GPU target for im2col and col2im
    _im2col_kernel.set_target(CLScheduler::get().target());
    _col2im_kernel.set_target(CLScheduler::get().target());

    const ICLTensor *gemm_input_to_use  = input;
    ICLTensor       *gemm_output_to_use = output;

    const ICLTensor *biases_to_use = (_append_bias && !_skip_im2col) ? biases : nullptr;

    // Get parameters from conv_info
    unsigned int stride_x = 0;
    unsigned int stride_y = 0;
    std::tie(stride_x, stride_y) = conv_info.stride();

    // Get convolved dimensions
    unsigned int conv_w = 0;
    unsigned int conv_h = 0;
    std::tie(conv_w, conv_h) = scaled_dimensions(input->info()->dimension(idx_width),
                                                 input->info()->dimension(idx_height),
                                                 kernel_width,
                                                 kernel_height,
                                                 conv_info,
                                                 dilation);

    unsigned int mat_weights_cols = weights->info()->dimension(idx_kernels) / num_groups;

    // _weights_reshaped will be auto configured in the kernel.
    // Just append biases and do not transpose 1xW as it will be reshaped in CLGEMM
    _reshape_weights.configure(weights, biases_to_use, &_weights_reshaped, num_groups);

    // Create tensor to store im2col reshaped inputs
    if(!_skip_im2col)
    {
        _memory_group.manage(&_im2col_output);

        // Configure and tune im2col. im2col output shape is auto-initialized
        _im2col_kernel.configure(input, &_im2col_output, Size2D(kernel_width, kernel_height), conv_info, _append_bias, dilation, num_groups);

        // Set quantization info
        _im2col_output.info()->set_quantization_info(input->info()->quantization_info());
        CLScheduler::get().tune_kernel_static(_im2col_kernel);

        // Update GEMM input
        gemm_input_to_use = &_im2col_output;
    }
    else if(_append_bias)
    {
        // Configure add bias kernel
        _add_bias_kernel.configure(ArithmeticOperation::ADD, output, biases, output, ConvertPolicy::SATURATE);
    }

    // Create GEMM output tensor
    if(!_skip_col2im)
    {
        TensorShape shape_gemm;

        // If we cannot skip col2im it means we run im2col as well
        shape_gemm = _im2col_output.info()->tensor_shape();
        shape_gemm.set(0, mat_weights_cols);
        shape_gemm.set(1, conv_w * conv_h);

        // FIXME: input->clone() doesn't work with subtensors for grouped convolutions.
        TensorInfo info_gemm(shape_gemm, 1, data_type);
        info_gemm.set_quantization_info(output->info()->quantization_info()).set_data_layout(input->info()->data_layout());
        _gemm_output.allocator()->init(info_gemm);
        _memory_group.manage(&_gemm_output);

        // Update GEMM output
        gemm_output_to_use = &_gemm_output;
    }

    GEMMLowpOutputStageInfo gemmlowp_output_stage;
    gemmlowp_output_stage.type                = GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
    gemmlowp_output_stage.gemmlowp_offset     = 0;
    gemmlowp_output_stage.gemmlowp_multiplier = 0;
    gemmlowp_output_stage.gemmlowp_shift      = 0;

    // Configure output stage for quantized case
    if(_is_quantized)
    {
        const QuantizationInfo output_quant_info = (output->info()->total_size() == 0) ? input->info()->quantization_info() : output->info()->quantization_info();

        const float multiplier        = (input->info()->quantization_info().scale * weights->info()->quantization_info().scale) / output_quant_info.scale;
        int         output_multiplier = 0;
        int         output_shift      = 0;
        quantization::calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift);

        int min_activation = 0;
        int max_activation = 0;

        const std::set<ActivationLayerInfo::ActivationFunction> supported_acts = { ActivationLayerInfo::ActivationFunction::RELU,
                                                                                   ActivationLayerInfo::ActivationFunction::BOUNDED_RELU,
                                                                                   ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU
                                                                                 };

        if(_is_activationlayer_enabled && supported_acts.count(act_info.activation()) != 0)
        {
            const int a_const_int = output_quant_info.quantize(act_info.a(), RoundingPolicy::TO_NEAREST_UP);
            const int b_const_int = output_quant_info.quantize(act_info.b(), RoundingPolicy::TO_NEAREST_UP);

            min_activation = act_info.activation() != ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU ? output_quant_info.offset : b_const_int;
            max_activation = act_info.activation() == ActivationLayerInfo::ActivationFunction::RELU ? 255 : a_const_int;

            // If the activation layer is RELU, BOUNDED_RELU or LU_BOUNDED_RELU, we can use the GEMMLowp output stage to perform this operation
            _is_activationlayer_enabled = false;
        }

        // Set the GEMMLowp output stage info
        gemmlowp_output_stage.gemmlowp_offset     = output_quant_info.offset;
        gemmlowp_output_stage.gemmlowp_multiplier = output_multiplier;
        gemmlowp_output_stage.gemmlowp_shift      = output_shift;
        gemmlowp_output_stage.gemmlowp_min_bound  = min_activation;
        gemmlowp_output_stage.gemmlowp_max_bound  = max_activation;
    }

    // Configure and tune GEMM
    // In case of NHWC, we need to run GEMM3D (gemm_3d_depth != 0) in order to avoid reshaping the output matrix
    const unsigned int gemm_3d_depth = (data_layout == DataLayout::NHWC) ? conv_h : 0;

    configure_mm(gemm_input_to_use, &_weights_reshaped, biases, gemm_output_to_use, gemmlowp_output_stage, gemm_3d_depth);

    if(!_skip_im2col)
    {
        _im2col_output.allocator()->allocate();
    }

    if(!_skip_col2im)
    {
        // Configure and tune Col2Im
        _col2im_kernel.configure(gemm_output_to_use, output, Size2D(conv_w, conv_h), num_groups);
        CLScheduler::get().tune_kernel_static(_col2im_kernel);
    }

    if(!_skip_col2im)
    {
        _gemm_output.allocator()->allocate();
    }

    ARM_COMPUTE_ERROR_ON_MSG((output->info()->dimension(idx_width) != conv_w) || (output->info()->dimension(idx_height) != conv_h),
                             "Output shape does not match the expected one");

    if(_is_activationlayer_enabled)
    {
        _activationlayer_function.configure(output, nullptr, act_info);
    }

    ARM_COMPUTE_UNUSED(weights_info);
}

Status CLGEMMConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                        const WeightsInfo &weights_info, const Size2D &dilation, const ActivationLayerInfo &act_info, unsigned int num_groups)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights_info.are_reshaped(), "Weights already reshaped are not supported!");
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, weights);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((num_groups != 1) && (input->data_layout() != DataLayout::NCHW), "Grouping (num_groups != 1) with NHWC data layout is not supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((num_groups != 1) && (input->data_type() == DataType::QASYMM8), "Grouping (num_groups != 1) is not supported with QASYMM8");
    ARM_COMPUTE_RETURN_ERROR_ON(((input->dimension(2) / weights->dimension(2)) != num_groups) && (input->data_layout() == DataLayout::NCHW));

    const DataLayout data_layout = input->data_layout();
    const DataType   data_type   = input->data_type();
    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        idx_channel = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);
    const int        idx_kernels = get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES);

    const unsigned int kernel_width  = weights->dimension(idx_width);
    const unsigned int kernel_height = weights->dimension(idx_height);

    TensorInfo         im2col_reshaped_info, info_gemm, weights_reshaped_info;
    const ITensorInfo *gemm_input_to_use  = input;
    const ITensorInfo *gemm_output_to_use = output;
    const ITensorInfo *weights_to_use     = weights;

    const bool is_quantized               = is_data_type_quantized_asymmetric(data_type);
    const bool append_bias                = (biases != nullptr) && (!is_quantized);
    const bool skip_im2col                = (data_layout == DataLayout::NHWC && kernel_width == 1 && kernel_height == 1 && conv_info.stride().first == 1 && conv_info.stride().second == 1);
    const bool skip_col2im                = data_layout == DataLayout::NHWC;
    bool       is_activationlayer_enabled = act_info.enabled();
    // In case of F16, fused bias will be used in GEMM
    const bool run_addition = (skip_im2col) && (append_bias) && (data_type != DataType::F16);

    ARM_COMPUTE_RETURN_ERROR_ON((weights->dimension(idx_channel) * num_groups) != input->dimension(idx_channel));
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
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
        }
        ARM_COMPUTE_RETURN_ERROR_ON(biases->dimension(0) != weights->dimension(idx_kernels));
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
    }

    if(act_info.enabled())
    {
        ARM_COMPUTE_ERROR_ON(act_info.b() > act_info.a());
    }

    // Get convolved dimensions
    unsigned int conv_w = 0;
    unsigned int conv_h = 0;

    std::tie(conv_w, conv_h) = scaled_dimensions(input->dimension(idx_width),
                                                 input->dimension(idx_height),
                                                 kernel_width,
                                                 kernel_height,
                                                 conv_info,
                                                 dilation);

    unsigned int mat_weights_cols = weights->dimension(idx_kernels) / num_groups;

    // Output tensor auto inizialitation if not yet initialized
    ARM_COMPUTE_RETURN_ON_ERROR(CLConvolutionLayerReshapeWeights::validate(weights, is_quantized ? nullptr : biases, nullptr, num_groups));
    weights_reshaped_info = TensorInfo(compute_weights_reshaped_shape(*weights, (append_bias && !skip_im2col), num_groups), 1, data_type);
    weights_to_use        = &weights_reshaped_info;

    if(!skip_im2col)
    {
        const Size2D kernel_dims(kernel_width, kernel_height);

        // Output tensor auto initialization if not yet initialized
        TensorShape expected_output_shape = compute_im2col_conv_shape(input, kernel_dims, conv_info, append_bias, dilation, num_groups == 1, num_groups);

        auto_init_if_empty(im2col_reshaped_info, input->clone()->set_tensor_shape(expected_output_shape));

        ARM_COMPUTE_RETURN_ON_ERROR(CLIm2ColKernel::validate(input, &im2col_reshaped_info, kernel_dims, conv_info, append_bias, dilation, num_groups));
        gemm_input_to_use = &im2col_reshaped_info;
    }
    else if(run_addition)
    {
        // Validate add bias kernel
        ARM_COMPUTE_RETURN_ON_ERROR(CLSaturatedArithmeticOperationKernel::validate(ArithmeticOperation::ADD, output, biases, output, ConvertPolicy::SATURATE));
    }

    // Create GEMM output tensor
    if(!skip_col2im)
    {
        TensorShape shape_gemm;

        shape_gemm = gemm_input_to_use->tensor_shape();
        shape_gemm.set(0, mat_weights_cols);
        shape_gemm.set(1, conv_w * conv_h);

        info_gemm = TensorInfo(shape_gemm, 1, data_type);
        info_gemm.set_quantization_info(output->quantization_info()).set_data_layout(input->data_layout());
        gemm_output_to_use = &info_gemm;
    }

    GEMMLowpOutputStageInfo gemmlowp_output_stage;
    gemmlowp_output_stage.type                = GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
    gemmlowp_output_stage.gemmlowp_offset     = 0;
    gemmlowp_output_stage.gemmlowp_multiplier = 0;
    gemmlowp_output_stage.gemmlowp_shift      = 0;

    if(is_quantized)
    {
        const QuantizationInfo output_quant_info = (output->total_size() == 0) ? input->quantization_info() : output->quantization_info();

        const float multiplier        = (input->quantization_info().scale * weights->quantization_info().scale) / output_quant_info.scale;
        int         output_multiplier = 0;
        int         output_shift      = 0;

        ARM_COMPUTE_RETURN_ON_ERROR(quantization::calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift));

        int min_activation = 0;
        int max_activation = 0;

        const std::set<ActivationLayerInfo::ActivationFunction> supported_acts = { ActivationLayerInfo::ActivationFunction::RELU,
                                                                                   ActivationLayerInfo::ActivationFunction::BOUNDED_RELU,
                                                                                   ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU
                                                                                 };

        if(is_activationlayer_enabled && supported_acts.count(act_info.activation()) != 0)
        {
            const int a_const_int = output_quant_info.quantize(act_info.a(), RoundingPolicy::TO_NEAREST_UP);
            const int b_const_int = output_quant_info.quantize(act_info.b(), RoundingPolicy::TO_NEAREST_UP);

            min_activation = act_info.activation() != ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU ? output_quant_info.offset : b_const_int;
            max_activation = act_info.activation() == ActivationLayerInfo::ActivationFunction::RELU ? 255 : a_const_int;

            // If the activation layer is RELU, BOUNDED_RELU or LU_BOUNDED_RELU, we can use the GEMMLowp output stage to perform this operation
            is_activationlayer_enabled = false;
        }

        // Set the GEMMLowp output stage info
        gemmlowp_output_stage.gemmlowp_offset     = output_quant_info.offset;
        gemmlowp_output_stage.gemmlowp_multiplier = output_multiplier;
        gemmlowp_output_stage.gemmlowp_shift      = output_shift;
        gemmlowp_output_stage.gemmlowp_min_bound  = min_activation;
        gemmlowp_output_stage.gemmlowp_max_bound  = max_activation;
    }

    // In case of NHWC, we need to run GEMM3D (gemm_3d_depth != 0) in order to avoid reshaping the output matrix
    const unsigned int gemm_3d_depth = (data_layout == DataLayout::NHWC) ? conv_h : 0;

    ARM_COMPUTE_RETURN_ON_ERROR(validate_mm(gemm_input_to_use, weights_to_use, biases, gemm_output_to_use, gemmlowp_output_stage, gemm_3d_depth, skip_im2col, run_addition));

    // Validate Col2Im
    if(!skip_col2im)
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLCol2ImKernel::validate(gemm_output_to_use, output, Size2D(conv_w, conv_h), num_groups));
    }

    //Validate Activation Layer
    if(is_activationlayer_enabled)
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLActivationLayer::validate(output, nullptr, act_info));
    }

    return Status{};
}

void CLGEMMConvolutionLayer::run()
{
    prepare();

    _memory_group.acquire();

    // Run im2col
    if(!_skip_im2col)
    {
        CLScheduler::get().enqueue(_im2col_kernel);
    }

    // Runs CLGEMM or CLGEMMLowpMatrixMultiplyCore functions
    if(_is_quantized)
    {
        // Run gemmlowp
        _mm_gemmlowp.run();
    }
    else
    {
        // Run gemm
        _mm_gemm.run();
    }

    if(_run_addition)
    {
        CLScheduler::get().enqueue(_add_bias_kernel);
    }

    // Reshape output matrix
    if(!_skip_col2im)
    {
        CLScheduler::get().enqueue(_col2im_kernel, false);
    }

    //Run Activation Layer if enabled
    if(_is_activationlayer_enabled)
    {
        _activationlayer_function.run();
    }

    _memory_group.release();
}

void CLGEMMConvolutionLayer::prepare()
{
    if(!_is_prepared)
    {
        ARM_COMPUTE_ERROR_ON(!_original_weights->is_used());

        // Run weights reshaping and mark original weights tensor as unused
        _weights_reshaped.allocator()->allocate();
        _reshape_weights.run();
        _original_weights->mark_as_unused();

        // Prepare GEMM
        _is_quantized ? _mm_gemmlowp.prepare() : _mm_gemm.prepare();
        if(!_weights_reshaped.is_used())
        {
            _weights_reshaped.allocator()->free();
        }

        CLScheduler::get().queue().finish();
        _is_prepared = true;
    }
}
