/*
 * Copyright (c) 2017-2018 ARM Limited.
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

void CLConvolutionLayerReshapeWeights::configure(const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output)
{
    // Perform validation step
    ARM_COMPUTE_ERROR_ON_NULLPTR(weights, output);
    ARM_COMPUTE_ERROR_THROW_ON(CLConvolutionLayerReshapeWeights::validate(weights->info(),
                                                                          (biases != nullptr) ? biases->info() : nullptr,
                                                                          output->info()));

    const bool       append_biases = (biases != nullptr) && !is_data_type_quantized_asymmetric(weights->info()->data_type());
    const ICLTensor *biases_to_use = (append_biases) ? biases : nullptr;

    _weights_reshape_kernel.configure(weights, biases_to_use, output);

    output->info()->set_quantization_info(weights->info()->quantization_info());
}

Status CLConvolutionLayerReshapeWeights::validate(const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output)
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

        CLWeightsReshapeKernel::validate(weights, biases, output);
    }

    return Status{};
}

void CLConvolutionLayerReshapeWeights::run()
{
    CLScheduler::get().enqueue(_weights_reshape_kernel);
}

CLGEMMConvolutionLayer::CLGEMMConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(memory_manager), _reshape_weights(), _im2col_kernel(), _mm_gemm(memory_manager), _mm_gemmlowp(memory_manager), _gemmlowp_output_stage(), _col2im_kernel(), _activationlayer_function(),
      _add_bias_kernel(), _reshape_layer(), _original_weights(nullptr), _im2col_output(), _weights_reshaped(), _gemm_output(), _tmp_output(), _data_layout(DataLayout::NCHW), _append_bias(false),
      _skip_im2col(false), _is_quantized(false), _is_activationlayer_enabled(false), _is_prepared(false)
{
}

void CLGEMMConvolutionLayer::configure_mm(const ICLTensor *input, const ICLTensor *weights, ICLTensor *output, int gemm_3d_depth)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights);
    ARM_COMPUTE_ERROR_THROW_ON(validate_mm(input->info(), weights->info(), output->info(), gemm_3d_depth, _skip_im2col));

    if(_is_quantized)
    {
        // Since we need negative offsets for computing convolution, we need to change QuantizationInfo()
        // Extract and negate input and weights offset
        const QuantizationInfo input_quantization_info   = input->info()->quantization_info();
        const QuantizationInfo weights_quantization_info = weights->info()->quantization_info();

        input->info()->set_quantization_info(QuantizationInfo(input_quantization_info.scale, -input_quantization_info.offset));
        weights->info()->set_quantization_info(QuantizationInfo(weights_quantization_info.scale, -weights_quantization_info.offset));

        _mm_gemmlowp.configure(input, weights, output, GEMMInfo(false, false, true /* Reshape weights only for the first run*/));

        // Revert back QuantizatioInfo as input and weights could be used in other convolution layers
        input->info()->set_quantization_info(input_quantization_info);
        weights->info()->set_quantization_info(weights_quantization_info);
    }
    else
    {
        // Configure matrix multiply function
        _mm_gemm.configure(input, weights, nullptr, output, 1.0f, 0.0f, GEMMInfo(false, false, true /* Reshape weights only for the first run*/, gemm_3d_depth,
                                                                                 _skip_im2col /* Reinterpret the input as 3D if im2col is skipped */));
    }
}

Status CLGEMMConvolutionLayer::validate_mm(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *output, int gemm_3d_depth, bool skip_im2col)
{
    const bool is_quantized = is_data_type_quantized_asymmetric(input->data_type());

    const GEMMInfo &gemm_info = GEMMInfo(false, false, true /* Reshape weights only for the first run */, gemm_3d_depth, skip_im2col /* Reinterpret the input as 3D if im2col is skipped */);
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
        return CLGEMMLowpMatrixMultiplyCore::validate(input_qa.get(), weights_qa.get(), output, gemm_info);
    }
    else
    {
        // Perform validation step on Matrix multiply function
        return CLGEMM::validate(input, weights, nullptr, output, 1.0f, 0.0f, gemm_info);
    }
}

void CLGEMMConvolutionLayer::configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info,
                                       const Size2D &dilation, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);

    ARM_COMPUTE_ERROR_THROW_ON(CLGEMMConvolutionLayer::validate(input->info(),
                                                                weights->info(),
                                                                biases != nullptr ? biases->info() : nullptr,
                                                                output->info(),
                                                                conv_info,
                                                                weights_info,
                                                                dilation,
                                                                act_info));

    const DataType   data_type   = input->info()->data_type();
    const DataLayout data_layout = input->info()->data_layout();
    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        idx_kernels = get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES);

    const unsigned int kernel_width  = weights->info()->dimension(idx_width);
    const unsigned int kernel_height = weights->info()->dimension(idx_height);

    _is_prepared      = weights_info.retain_internal_weights();
    _original_weights = weights;
    _is_quantized     = is_data_type_quantized_asymmetric(input->info()->data_type());
    _data_layout      = data_layout;
    _skip_im2col      = (data_layout == DataLayout::NHWC && kernel_width == 1 && kernel_height == 1 && conv_info.stride().first == 1 && conv_info.stride().second == 1) && !_is_quantized;
    _append_bias      = (biases != nullptr) && (!_is_quantized);

    // Set the GPU target for im2col and col2im
    _im2col_kernel.set_target(CLScheduler::get().target());
    _col2im_kernel.set_target(CLScheduler::get().target());

    bool             is_nhwc                   = _data_layout == DataLayout::NHWC;
    const ICLTensor *gemm_input_to_use         = input;
    ICLTensor       *gemm_output_to_use        = output;
    ICLTensor       *gemm_output_staged_to_use = output;

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

    unsigned int mat_weights_cols = weights->info()->dimension(idx_kernels);

    // _weights_reshaped will be auto configured in the kernel.
    // Just append biases and do not transpose 1xW as it will be reshaped in CLGEMM
    _reshape_weights.configure(weights, biases_to_use, &_weights_reshaped);

    weights = &_weights_reshaped;

    // Create tensor to store im2col reshaped inputs
    if(!_skip_im2col)
    {
        _memory_group.manage(&_im2col_output);

        // Configure and tune im2col. im2col output shape is auto-initialized
        _im2col_kernel.configure(input, &_im2col_output, Size2D(kernel_width, kernel_height), conv_info, _append_bias, dilation);

        // Set quantization info
        _im2col_output.info()->set_quantization_info(input->info()->quantization_info());
        CLScheduler::get().tune_kernel_static(_im2col_kernel);

        // Update GEMM input
        gemm_input_to_use = &_im2col_output;
    }
    else if(_append_bias)
    {
        // Configure add bias kernel
        _add_bias_kernel.configure(output, biases, output, ConvertPolicy::SATURATE);
    }

    // Create GEMM output tensor
    if(!is_nhwc || _is_quantized)
    {
        // Calculate GEMM output shape
        TensorShape shape_gemm = _im2col_output.info()->tensor_shape();
        shape_gemm.set(0, mat_weights_cols);
        shape_gemm.set(1, conv_w * conv_h);

        // GEMM output should be S32 for acquiring raw integer accumulator without quantized postprocessing for quantized asymmetric input.
        const DataType gemm_data_type = _is_quantized ? DataType::S32 : data_type;
        // FIXME: input->clone() doesn't work with subtensors for grouped convolutions.
        TensorInfo info_gemm(shape_gemm, 1, gemm_data_type);
        info_gemm.set_quantization_info(output->info()->quantization_info());
        _gemm_output.allocator()->init(info_gemm);
        _memory_group.manage(&_gemm_output);

        // Update GEMM output
        gemm_output_to_use = &_gemm_output;
    }

    // Configure and tune GEMM
    configure_mm(gemm_input_to_use, weights, gemm_output_to_use, (data_layout == DataLayout::NHWC) ? conv_h : 1);

    if(!_skip_im2col)
    {
        _im2col_output.allocator()->allocate();
    }

    // Configure output stage for quantized case
    if(_is_quantized)
    {
        const QuantizationInfo output_quant_info = (output->info()->total_size() == 0) ? input->info()->quantization_info() : output->info()->quantization_info();

        float multiplier = input->info()->quantization_info().scale * weights->info()->quantization_info().scale / output_quant_info.scale;
        int   output_multiplier, output_shift;
        quantization::calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift);

        _memory_group.manage(&_tmp_output);
        gemm_output_staged_to_use = &_tmp_output;

        _gemmlowp_output_stage.configure(gemm_output_to_use, biases, gemm_output_staged_to_use, output_multiplier, output_shift, output_quant_info.offset);
    }

    if(!is_nhwc || _is_quantized)
    {
        if(input->info()->data_layout() == DataLayout::NCHW)
        {
            // Configure and tune Col2Im
            _col2im_kernel.configure(_is_quantized ? gemm_output_staged_to_use : gemm_output_to_use, output, std::make_pair(conv_w, conv_h));
            CLScheduler::get().tune_kernel_static(_col2im_kernel);
        }
        else
        {
            // Configure reshape layer
            _reshape_layer.configure(_is_quantized ? gemm_output_staged_to_use : gemm_output_to_use, output);
        }
    }

    if(!is_nhwc || _is_quantized)
    {
        _tmp_output.allocator()->allocate();
        _gemm_output.allocator()->allocate();
    }

    ARM_COMPUTE_ERROR_ON_MSG((output->info()->dimension(idx_width) != conv_w) || (output->info()->dimension(idx_height) != conv_h),
                             "Output shape does not match the expected one");

    //Configure Activation Layer
    _is_activationlayer_enabled = act_info.enabled();

    if(_is_activationlayer_enabled)
    {
        _activationlayer_function.configure(output, nullptr, act_info);
    }

    ARM_COMPUTE_UNUSED(weights_info);
}

Status CLGEMMConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                        const WeightsInfo &weights_info, const Size2D &dilation, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights_info.are_reshaped(), "Weights already reshaped are not supported!");
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, weights);

    const DataLayout data_layout = input->data_layout();
    const DataType   data_type   = input->data_type();
    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        idx_channel = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);
    const int        idx_kernels = get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES);

    const unsigned int kernel_width  = weights->dimension(idx_width);
    const unsigned int kernel_height = weights->dimension(idx_height);

    TensorInfo         im2col_reshaped_info, info_gemm, tmp_info, weights_reshaped_info;
    const ITensorInfo *gemm_input_to_use         = input;
    const ITensorInfo *gemm_output_to_use        = output;
    const ITensorInfo *gemm_output_staged_to_use = output;
    const ITensorInfo *weights_to_use            = weights;

    const bool is_nhwc      = data_layout == DataLayout::NHWC;
    const bool is_quantized = is_data_type_quantized_asymmetric(data_type);
    const bool skip_im2col  = (data_layout == DataLayout::NHWC && kernel_width == 1 && kernel_height == 1 && conv_info.stride().first == 1 && conv_info.stride().second == 1) && !is_quantized;
    const bool append_bias  = (biases != nullptr) && (!is_quantized);

    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(idx_channel) != input->dimension(idx_channel));
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

    unsigned int mat_weights_cols = weights->dimension(idx_kernels);

    // Output tensor auto inizialitation if not yet initialized
    ARM_COMPUTE_RETURN_ON_ERROR(CLConvolutionLayerReshapeWeights::validate(weights, is_quantized ? nullptr : biases, nullptr));
    weights_reshaped_info = TensorInfo(compute_weights_reshaped_shape(*weights, (append_bias && !skip_im2col)), 1, data_type);
    weights_to_use        = &weights_reshaped_info;

    if(!skip_im2col)
    {
        const Size2D kernel_dims(kernel_width, kernel_height);

        // Output tensor auto initialization if not yet initialized
        TensorShape expected_output_shape = compute_im2col_conv_shape(input, kernel_dims, conv_info, append_bias, dilation, true /* num_groups == 1, num_groups */);

        auto_init_if_empty(im2col_reshaped_info, input->clone()->set_tensor_shape(expected_output_shape));

        ARM_COMPUTE_RETURN_ON_ERROR(CLIm2ColKernel::validate(input, &im2col_reshaped_info, kernel_dims, conv_info, append_bias, dilation));
        gemm_input_to_use = &im2col_reshaped_info;
    }
    else if(append_bias)
    {
        // Validate add bias kernel
        ARM_COMPUTE_RETURN_ON_ERROR(CLArithmeticAdditionKernel::validate(output, biases, output, ConvertPolicy::SATURATE));
    }

    // Create GEMM output tensor
    if(!is_nhwc || is_quantized)
    {
        TensorShape shape_gemm = gemm_input_to_use->tensor_shape();
        shape_gemm.set(0, mat_weights_cols);
        shape_gemm.set(1, conv_w * conv_h);
        const DataType gemm_data_type = is_quantized ? DataType::S32 : data_type;
        // GEMM output should be S32 for acquiring raw integer accumulator without quantized postprocessing for quantized asymmetric input.
        info_gemm = TensorInfo(shape_gemm, 1, gemm_data_type);
        info_gemm.set_quantization_info(output->quantization_info());
        gemm_output_to_use = &info_gemm;
    }

    ARM_COMPUTE_RETURN_ON_ERROR(validate_mm(gemm_input_to_use, weights_to_use, gemm_output_to_use, (data_layout == DataLayout::NHWC) ? conv_h : 1, skip_im2col));

    if(is_quantized)
    {
        float multiplier = input->quantization_info().scale * weights_to_use->quantization_info().scale / output->quantization_info().scale;
        int   output_multiplier, output_shift;
        quantization::calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift);

        tmp_info = TensorInfo(gemm_output_to_use->tensor_shape(), 1, DataType::QASYMM8);
        tmp_info.set_quantization_info(output->quantization_info());
        gemm_output_staged_to_use = &tmp_info;

        // Validate output stage for quantized case
        CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint::validate(gemm_output_to_use, biases, gemm_output_staged_to_use, output->quantization_info().offset);
    }

    // Validate Col2Im
    if(!is_nhwc || is_quantized)
    {
        if(input->data_layout() == DataLayout::NCHW)
        {
            ARM_COMPUTE_RETURN_ON_ERROR(CLCol2ImKernel::validate(is_quantized ? gemm_output_staged_to_use : gemm_output_to_use,
                                                                 output,
                                                                 std::make_pair(conv_w, conv_h)));
        }
    }

    //Validate Activation Layer
    if(act_info.enabled())
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

        // Run output stage
        _gemmlowp_output_stage.run();
    }
    else
    {
        // Run gemm
        _mm_gemm.run();
    }

    if(_skip_im2col && _append_bias)
    {
        CLScheduler::get().enqueue(_add_bias_kernel);
    }

    // Reshape output matrix
    if(_data_layout == DataLayout::NCHW || _is_quantized)
    {
        if(_data_layout == DataLayout::NCHW)
        {
            CLScheduler::get().enqueue(_col2im_kernel, false);
        }
        else
        {
            _reshape_layer.run();
        }
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
