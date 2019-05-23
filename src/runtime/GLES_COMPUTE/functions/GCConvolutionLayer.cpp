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

#include "arm_compute/runtime/GLES_COMPUTE/functions/GCConvolutionLayer.h"

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCScheduler.h"

#include <cmath>
#include <memory>
#include <tuple>

using namespace arm_compute;

GCConvolutionLayerReshapeWeights::GCConvolutionLayerReshapeWeights()
    : _weights_reshape_kernel()
{
}

void GCConvolutionLayerReshapeWeights::configure(const IGCTensor *weights, const IGCTensor *biases, IGCTensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(weights, output);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON(weights->info()->num_dimensions() > 4);

    if(biases != nullptr)
    {
        ARM_COMPUTE_ERROR_ON(is_data_type_quantized_asymmetric(weights->info()->data_type()));
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(weights, biases);
        ARM_COMPUTE_ERROR_ON(biases->info()->dimension(0) != weights->info()->dimension(3));
        ARM_COMPUTE_ERROR_ON(biases->info()->num_dimensions() > 1);
    }

    const bool       append_biases = (biases != nullptr) && !is_data_type_quantized_asymmetric(weights->info()->data_type());
    const IGCTensor *biases_to_use = (append_biases) ? biases : nullptr;

    _weights_reshape_kernel.configure(weights, biases_to_use, output);
}

void GCConvolutionLayerReshapeWeights::run()
{
    GCScheduler::get().dispatch(_weights_reshape_kernel);
}

GCConvolutionLayer::GCConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _reshape_weights(), _input_im2col_kernel(), _mm_gemm(), _output_col2im_kernel(), _fill_border(), _activationlayer_function(), _original_weights(nullptr),
      _input_im2col_reshaped(), _input_interleaved_reshaped(), _weights_reshaped(), _weights_transposed(), _gemm_output(), _tmp_output(), _is_activationlayer_enabled(false), _is_prepared(false)
{
}

void GCConvolutionLayer::configure_mm(const IGCTensor *input, const IGCTensor *weights, IGCTensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights);
    ARM_COMPUTE_ERROR_THROW_ON(validate_mm(input->info(), weights->info(), output->info()));

    _mm_gemm.configure(input, weights, nullptr, output, 1.f, 0.0f, GEMMInfo(false, false, true /* Reshape weights only for the first run */));
}

Status GCConvolutionLayer::validate_mm(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *output)
{
    // Perform validation step on Matrix multiply function
    GCGEMM::validate(input, weights, nullptr, output, 1.0f, 0.0f, GEMMInfo(false, false, true /* Reshape weights only for the first run */));
    return Status{};
}

void GCConvolutionLayer::configure(const IGCTensor *input, const IGCTensor *weights, const IGCTensor *biases, IGCTensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info,
                                   const Size2D &dilation, const ActivationLayerInfo &act_info, unsigned int num_groups)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    ARM_COMPUTE_ERROR_ON_MSG(weights_info.are_reshaped(), "Weights already reshaped are not supported!");
    ARM_COMPUTE_ERROR_ON(weights->info()->dimension(2) != input->info()->dimension(2));
    ARM_COMPUTE_ERROR_ON(weights->info()->num_dimensions() > 4);
    ARM_COMPUTE_ERROR_ON(num_groups > 1);
    ARM_COMPUTE_UNUSED(num_groups);

    _is_prepared      = false;
    _original_weights = weights;

    if(biases != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
        ARM_COMPUTE_ERROR_ON(biases->info()->dimension(0) != weights->info()->dimension(3));
        ARM_COMPUTE_ERROR_ON(biases->info()->num_dimensions() > 1);
    }

    const DataType dt = input->info()->data_type();

    // Set the GPU target for im2col and col2im
    _input_im2col_kernel.set_target(GCScheduler::get().get_target());
    _output_col2im_kernel.set_target(GCScheduler::get().get_target());

    const bool       append_bias   = (biases != nullptr);
    const unsigned   bias_element  = (append_bias) ? 1 : 0;
    const IGCTensor *biases_to_use = (append_bias) ? biases : nullptr;

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
    // Just append biases and do not transpose 1xW as it will be reshaped in GCGEMM
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
    TensorInfo im2col_reshaped_info(shape_im2col, 1, dt);
    _input_im2col_reshaped.allocator()->init(im2col_reshaped_info);
    _memory_group.manage(&_input_im2col_reshaped);

    // Create GEMM output tensor
    TensorShape shape_gemm = _input_im2col_reshaped.info()->tensor_shape();
    shape_gemm.set(0, mat_weights_cols);
    shape_gemm.set(1, mat_input_rows);
    const DataType gemm_data_type = dt;

    // FIXME: input->clone() doesn't work with subtensors for grouped convolutions.
    TensorInfo info_gemm(shape_gemm, 1, gemm_data_type);
    _gemm_output.allocator()->init(info_gemm);
    _memory_group.manage(&_gemm_output);

    if(dt == DataType::F16)
    {
        BorderSize border_size = BorderSize(conv_info.pad_top(), conv_info.pad_right(), conv_info.pad_bottom(), conv_info.pad_left());
        input->info()->extend_padding(border_size);
        _fill_border.configure(input, border_size, BorderMode::CONSTANT, PixelValue()); // for PAD of im2col fp16: consider it as border
    }
    // Configure im2col
    _input_im2col_kernel.configure(input, &_input_im2col_reshaped, Size2D(kernel_width, kernel_height), conv_info, append_bias, dilation);

    // Configure GEMM
    configure_mm(&_input_im2col_reshaped, weights, &_gemm_output);

    _input_im2col_reshaped.allocator()->allocate();

    // Configure Col2Im
    _output_col2im_kernel.configure(&_gemm_output, output, std::make_pair(conv_w, conv_h));
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

void GCConvolutionLayer::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_memory_group);

    // Run im2col
    GCScheduler::get().dispatch(_fill_border);
    GCScheduler::get().memory_barrier();
    GCScheduler::get().dispatch(_input_im2col_kernel);

    // Run gemm on reshaped matrices
    _mm_gemm.run();
    GCScheduler::get().memory_barrier();

    // Reshape output matrix
    GCScheduler::get().dispatch(_output_col2im_kernel, false);
    GCScheduler::get().memory_barrier();

    // Run Activation Layer
    if(_is_activationlayer_enabled)
    {
        _activationlayer_function.run();
    }
}

void GCConvolutionLayer::prepare()
{
    if(!_is_prepared)
    {
        ARM_COMPUTE_ERROR_ON(!_original_weights->is_used());

        // Run weights reshaping and mark as unused
        _weights_reshaped.allocator()->allocate();
        _reshape_weights.run();

        // Mark original weights tensor as unused
        _original_weights->mark_as_unused();

        _is_prepared = true;
    }
}
