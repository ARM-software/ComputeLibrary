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
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::QS8, DataType::QASYMM8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 4);

    if(biases != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(is_data_type_quantized_asymmetric(weights->data_type()));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(weights, biases);
        ARM_COMPUTE_RETURN_ERROR_ON(biases->dimension(0) != weights->dimension(3));
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
    }

    if((output != nullptr) && (output->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(weights, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT(weights, output);

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
      _original_weights(nullptr), _im2col_output(), _weights_reshaped(), _gemm_output(), _tmp_output(), _is_quantized(false), _is_activationlayer_enabled(false), _is_prepared(false),
      _retain_internal_weights(false)
{
}

void CLGEMMConvolutionLayer::configure_mm(const ICLTensor *input, const ICLTensor *weights, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights);
    ARM_COMPUTE_ERROR_THROW_ON(validate_mm(input->info(), weights->info(), output->info()));

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
        _mm_gemm.configure(input, weights, nullptr, output, 1.0f, 0.0f, GEMMInfo(false, false, true /* Reshape weights only for the first run*/));
    }
}

Status CLGEMMConvolutionLayer::validate_mm(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *output)
{
    const bool is_quantized = is_data_type_quantized_asymmetric(input->data_type());

    const GEMMInfo &gemm_info = GEMMInfo(false, false, true /* Reshape weights only for the first run */);
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
        CLGEMMLowpMatrixMultiplyCore::validate(input_qa.get(), weights_qa.get(), output, gemm_info);
    }
    else
    {
        // Perform validation step on Matrix multiply function
        CLGEMM::validate(input, weights, nullptr, output, 1.0f, 0.0f, gemm_info);
    }
    return Status{};
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

    _is_prepared             = false;
    _original_weights        = weights;
    _is_quantized            = is_data_type_quantized_asymmetric(input->info()->data_type());
    _retain_internal_weights = weights_info.retain_internal_weights();

    const DataType dt = input->info()->data_type();

    // Set the GPU target for im2col and col2im
    _im2col_kernel.set_target(CLScheduler::get().target());
    _col2im_kernel.set_target(CLScheduler::get().target());

    const bool append_bias = (biases != nullptr) && (!_is_quantized);

    const unsigned   bias_element  = (append_bias) ? 1 : 0;
    const ICLTensor *biases_to_use = (append_bias) ? biases : nullptr;

    // Get parameters from conv_info
    unsigned int stride_x = 0;
    unsigned int stride_y = 0;
    std::tie(stride_x, stride_y) = conv_info.stride();

    // Get convolved dimensions
    unsigned int conv_w = 0;
    unsigned int conv_h = 0;

    const unsigned int kernel_width  = weights->info()->dimension(0);
    const unsigned int kernel_height = weights->info()->dimension(1);
    std::tie(conv_w, conv_h) = scaled_dimensions(input->info()->dimension(0), input->info()->dimension(1), kernel_width, kernel_height,
                                                 conv_info, dilation);

    unsigned int mat_weights_cols = weights->info()->dimension(3);
    unsigned int mat_weights_rows = weights->info()->dimension(0) * weights->info()->dimension(1) * weights->info()->dimension(2) + bias_element;

    // _weights_reshaped will be auto configured in the kernel.
    // Just append biases and do not transpose 1xW as it will be reshaped in CLGEMM
    _reshape_weights.configure(weights, biases_to_use, &_weights_reshaped);

    weights = &_weights_reshaped;

    // Create tensor to store im2col reshaped inputs
    const unsigned int mat_input_cols = mat_weights_rows;
    const unsigned int mat_input_rows = conv_w * conv_h;
    TensorShape        shape_im2col   = input->info()->tensor_shape();
    shape_im2col.set(0, mat_input_cols);
    shape_im2col.set(1, mat_input_rows);
    shape_im2col.set(2, 1);
    // FIXME: input->clone() doesn't work with subtensors for grouped convolutions.
    TensorInfo im2col_reshaped_info(shape_im2col, 1, dt, input->info()->fixed_point_position());
    im2col_reshaped_info.set_quantization_info(input->info()->quantization_info());
    _im2col_output.allocator()->init(im2col_reshaped_info);
    _memory_group.manage(&_im2col_output);

    // Create GEMM output tensor
    TensorShape shape_gemm = _im2col_output.info()->tensor_shape();
    shape_gemm.set(0, mat_weights_cols);
    shape_gemm.set(1, mat_input_rows);
    const DataType gemm_data_type = _is_quantized ? DataType::S32 : dt;
    // GEMM output should be S32 for acquiring raw integer accumulator without quantized postprocessing for quantized asymmetric input.
    // FIXME: input->clone() doesn't work with subtensors for grouped convolutions.
    TensorInfo info_gemm(shape_gemm, 1, gemm_data_type, input->info()->fixed_point_position());
    info_gemm.set_quantization_info(output->info()->quantization_info());
    _gemm_output.allocator()->init(info_gemm);
    _memory_group.manage(&_gemm_output);

    // Configure and tune im2col
    _im2col_kernel.configure(input, &_im2col_output, Size2D(kernel_width, kernel_height), conv_info, append_bias, dilation);
    CLScheduler::get().tune_kernel_static(_im2col_kernel);

    // Configure and tune GEMM
    configure_mm(&_im2col_output, weights, &_gemm_output);

    _im2col_output.allocator()->allocate();

    // Configure output stage for quantized case
    if(_is_quantized)
    {
        const QuantizationInfo output_quant_info = (output->info()->total_size() == 0) ? input->info()->quantization_info() : output->info()->quantization_info();

        float multiplier = input->info()->quantization_info().scale * weights->info()->quantization_info().scale / output_quant_info.scale;
        int   output_multiplier, output_shift;
        quantization::calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift);
        _memory_group.manage(&_tmp_output);
        _gemmlowp_output_stage.configure(&_gemm_output, biases, &_tmp_output, output_multiplier, output_shift, output_quant_info.offset);
    }

    // Configure and tune Col2Im
    _col2im_kernel.configure(_is_quantized ? &_tmp_output : &_gemm_output, output, std::make_pair(conv_w, conv_h));
    CLScheduler::get().tune_kernel_static(_col2im_kernel);
    if(_is_quantized)
    {
        _tmp_output.allocator()->allocate();
    }
    _gemm_output.allocator()->allocate();

    ARM_COMPUTE_ERROR_ON_MSG((output->info()->dimension(0) != conv_w) || (output->info()->dimension(1) != conv_h), "Output shape does not match the expected one");

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
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QASYMM8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT(input, weights);
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(2) != input->dimension(2));
    ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 4);

    if(act_info.enabled())
    {
        ARM_COMPUTE_ERROR_ON(act_info.b() > act_info.a());
    }

    const bool     is_quantized = is_data_type_quantized_asymmetric(input->data_type());
    const bool     append_bias  = (biases != nullptr) && (!is_quantized);
    const unsigned bias_element = (append_bias) ? 1 : 0;
    const DataType dt           = input->data_type();

    // Get convolved dimensions
    unsigned int conv_w = 0;
    unsigned int conv_h = 0;

    const unsigned int kernel_width  = weights->dimension(0);
    const unsigned int kernel_height = weights->dimension(1);

    std::tie(conv_w, conv_h) = scaled_dimensions(input->dimension(0), input->dimension(1), kernel_width, kernel_height, conv_info, dilation);

    unsigned int mat_weights_cols = weights->dimension(3);
    unsigned int mat_weights_rows = weights->dimension(0) * weights->dimension(1) * weights->dimension(2) + bias_element;

    ARM_COMPUTE_RETURN_ON_ERROR(CLConvolutionLayerReshapeWeights::validate(weights, is_quantized ? nullptr : biases, nullptr));

    // Create tensor info for im2col reshaped inputs
    const unsigned int mat_input_cols = mat_weights_rows;
    const unsigned int mat_input_rows = conv_w * conv_h;
    TensorShape        shape_im2col   = input->tensor_shape();
    shape_im2col.set(0, mat_input_cols);
    shape_im2col.set(1, mat_input_rows);
    shape_im2col.set(2, 1);
    TensorInfo im2col_reshaped_info(shape_im2col, 1, dt, input->fixed_point_position());
    im2col_reshaped_info.set_quantization_info(input->quantization_info());
    ARM_COMPUTE_RETURN_ON_ERROR(CLIm2ColKernel::validate(input, &im2col_reshaped_info, Size2D(kernel_width, kernel_height), conv_info, append_bias, dilation));

    // Create GEMM output tensor
    TensorShape shape_gemm = im2col_reshaped_info.tensor_shape();
    shape_gemm.set(0, mat_weights_cols);
    shape_gemm.set(1, mat_input_rows);
    const DataType gemm_data_type = is_quantized ? DataType::S32 : dt;
    // GEMM output should be S32 for acquiring raw integer accumulator without quantized postprocessing for quantized asymmetric input.
    TensorInfo info_gemm(shape_gemm, 1, gemm_data_type, input->fixed_point_position());
    info_gemm.set_quantization_info(output->quantization_info());

    ARM_COMPUTE_RETURN_ON_ERROR(validate_mm(&im2col_reshaped_info, weights, &info_gemm));
    TensorInfo tmp_info(shape_gemm, 1, DataType::QASYMM8, input->fixed_point_position());
    tmp_info.set_quantization_info(output->quantization_info());

    if(is_quantized)
    {
        float multiplier = input->quantization_info().scale * weights->quantization_info().scale / output->quantization_info().scale;
        int   output_multiplier, output_shift;
        quantization::calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift);
        // Validate output stage for quantized case
        CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint::validate(&info_gemm, biases, &tmp_info, output->quantization_info().offset);
    }

    // Validate Col2Im
    ARM_COMPUTE_RETURN_ON_ERROR(CLCol2ImKernel::validate(is_quantized ? &tmp_info : &info_gemm, output, std::make_pair(conv_w, conv_h)));

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
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT(input, biases);
        ARM_COMPUTE_RETURN_ERROR_ON(biases->dimension(0) != weights->dimension(3));
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
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
    CLScheduler::get().enqueue(_im2col_kernel);

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

    // Reshape output matrix
    CLScheduler::get().enqueue(_col2im_kernel, false);

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
        if(!_retain_internal_weights)
        {
            // Run weights reshaping and mark as unused
            ARM_COMPUTE_ERROR_ON(!_original_weights->is_used());
            _weights_reshaped.allocator()->allocate();
            _reshape_weights.run();
            _original_weights->mark_as_unused();
        }

        // Run GEMM prepare
        if(!_is_quantized)
        {
            _mm_gemm.prepare();
            if(!_weights_reshaped.is_used() && !_retain_internal_weights)
            {
                _weights_reshaped.allocator()->free();
            }
        }

        CLScheduler::get().queue().finish();
        _is_prepared = true;
    }
}
