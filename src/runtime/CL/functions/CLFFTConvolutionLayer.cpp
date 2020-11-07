/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLFFTConvolutionLayer.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CPP/CPPScheduler.h"
#include "src/core/CL/kernels/CLCopyKernel.h"
#include "src/core/CL/kernels/CLFFTDigitReverseKernel.h"
#include "src/core/CL/kernels/CLFFTRadixStageKernel.h"
#include "src/core/CL/kernels/CLFFTScaleKernel.h"
#include "src/core/CL/kernels/CLFillBorderKernel.h"
#include "src/core/CL/kernels/CLPadLayerKernel.h"
#include "src/core/CL/kernels/CLReductionOperationKernel.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/utils/helpers/fft.h"

#include "support/MemorySupport.h"

namespace arm_compute
{
namespace
{
int pad_decomposable(int N)
{
    const auto supported_radix = CLFFTRadixStageKernel::supported_radix();

    int  pad           = 0;
    bool is_decomposed = false;
    while(!is_decomposed)
    {
        const auto decomposed_vector = arm_compute::helpers::fft::decompose_stages(N++, supported_radix);
        is_decomposed                = !decomposed_vector.empty();
        if(!is_decomposed)
        {
            ++pad;
        }
    }
    return pad;
}
} // namespace
CLFFTConvolutionLayer::CLFFTConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(memory_manager),
      _flip_weights_func(),
      _permute_input_func(),
      _permute_output_func(),
      _permute_weights_func(),
      _permute_bias_func(),
      _pad_input_func(),
      _pad_weights_func(),
      _transform_input_func(memory_manager),
      _transform_weights_func(),
      _itransform_output_func(memory_manager),
      _prod_func(),
      _reduce_func(),
      _extract_output_func(),
      _bias_add_func(),
      _activation_layer_func(),
      _permuted_input(),
      _permuted_weights(),
      _permuted_bias(),
      _permuted_output(),
      _padded_input(),
      _padded_weights(),
      _flip_axis(),
      _flipped_weights(),
      _transformed_input(),
      _transformed_weights(),
      _input_weights_product(),
      _output_product(),
      _output_reduced(),
      _itransformed_output(),
      _reshaped_output(),
      _bias_output(),
      _original_weights(nullptr),
      _original_bias(nullptr),
      _is_activationlayer_enabled(false),
      _needs_permute(false),
      _has_bias(false),
      _is_prepared(false)
{
}

void CLFFTConvolutionLayer::configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                                      const ActivationLayerInfo &act_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, weights, biases, output, conv_info, act_info);
}

void CLFFTConvolutionLayer::configure(const CLCompileContext &compile_context, ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                                      const ActivationLayerInfo &act_info)
{
    _original_weights = weights;
    _original_bias    = biases;

    // Flat if bias addition is required
    _has_bias = biases != nullptr;

    // Get indices for the width and height
    const size_t idx_width  = get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_height = get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::HEIGHT);

    // Input shape, kernel size and output tile
    const Size2D input_dims  = Size2D(input->info()->tensor_shape()[idx_width], input->info()->tensor_shape()[idx_height]);
    const Size2D kernel_size = Size2D(weights->info()->tensor_shape()[idx_width], weights->info()->tensor_shape()[idx_height]);
    const Size2D pad_valid   = Size2D(pad_decomposable(input_dims.x() + kernel_size.x() - 1),
                                      pad_decomposable(input_dims.y() + kernel_size.y() - 1));
    // Tensors to use
    ICLTensor       *input_to_use   = input;
    const ICLTensor *weights_to_use = weights;
    ICLTensor       *output_to_use  = _has_bias ? &_bias_output : output;

    // Permute bias
    if(biases != nullptr)
    {
        _permute_bias_func.configure(compile_context, biases, &_permuted_bias, PermutationVector(1U, 2U, 0U));
        _permuted_bias.info()->set_data_layout(DataLayout::NCHW);
    }

    // Permute input if needed
    _needs_permute = input->info()->data_layout() == DataLayout::NHWC;
    if(_needs_permute)
    {
        _memory_group.manage(&_permuted_input);
        // Configure the function to transform the input tensor from NHWC -> NCHW
        _permute_input_func.configure(compile_context, input, &_permuted_input, PermutationVector(1U, 2U, 0U));
        _permuted_input.info()->set_data_layout(DataLayout::NCHW);

        // Configure the function to transform the weights tensor from HWI -> IHW
        _permute_weights_func.configure(compile_context, weights, &_permuted_weights, PermutationVector(1U, 2U, 0U));
        _permuted_weights.info()->set_data_layout(DataLayout::NCHW);

        input_to_use   = &_permuted_input;
        weights_to_use = &_permuted_weights;
    }

    // Flip weights
    _flipped_weights.allocator()->init(weights_to_use->info()->clone()->set_is_resizable(true).reset_padding());
    _flip_axis.allocator()->init(TensorInfo(TensorShape(2U), 1, DataType::U32));
    _flip_weights_func.configure(compile_context, weights_to_use, &_flipped_weights, &_flip_axis);

    // Pad weights
    const PaddingList padding_w = { { 0, input_dims.x() + pad_valid.x() - 1 }, { 0, input_dims.y() + pad_valid.y() - 1 } };
    _pad_weights_func.configure(compile_context, &_flipped_weights, &_padded_weights, padding_w);

    // Transform weights
    _transform_weights_func = support::cpp14::make_unique<CLFFT2D>();
    _transform_weights_func->configure(compile_context, &_padded_weights, &_transformed_weights, FFT2DInfo());

    // Pad input
    const PaddingList padding_in = { { 0, kernel_size.x() + pad_valid.x() - 1 }, { 0, kernel_size.y() + pad_valid.y() - 1 } };
    _memory_group.manage(&_padded_input);
    _pad_input_func.configure(compile_context, input_to_use, &_padded_input, padding_in);
    if(_needs_permute)
    {
        _permuted_input.allocator()->allocate();
    }

    // Transform input
    _memory_group.manage(&_transformed_input);
    _transform_input_func.configure(compile_context, &_padded_input, &_transformed_input, FFT2DInfo());
    _padded_input.allocator()->allocate();

    // Perform product
    _memory_group.manage(&_output_product);
    _prod_func.configure(compile_context, &_transformed_input, &_transformed_weights, &_output_product);
    _transformed_input.allocator()->allocate();

    // Perform reduction
    _memory_group.manage(&_output_reduced);
    _reduce_func.configure(compile_context, &_output_product, &_output_reduced, 2, ReductionOperation::SUM);
    _output_product.allocator()->allocate();

    // Transform output
    _memory_group.manage(&_itransformed_output);
    FFT2DInfo itranform_info;
    itranform_info.direction = FFTDirection::Inverse;
    _itransformed_output.allocator()->init(_output_reduced.info()->clone()->set_is_resizable(true).set_num_channels(1).reset_padding());
    _itransform_output_func.configure(compile_context, &_output_reduced, &_itransformed_output, itranform_info);
    _output_reduced.allocator()->allocate();

    // Reshape output
    TensorShape reshaped_shape = _itransformed_output.info()->tensor_shape();
    reshaped_shape.remove_dimension(2);
    _reshaped_output.allocator()->init(_itransformed_output.info()->clone()->set_tensor_shape(reshaped_shape));

    // Extract correct region
    const int start_left = kernel_size.x() - conv_info.pad_left() - 1;
    const int start_top  = kernel_size.y() - conv_info.pad_top() - 1;
    const int end_right  = _reshaped_output.info()->tensor_shape().x() - (kernel_size.x() - conv_info.pad_right() - 1) - pad_valid.x();
    const int end_botton = _reshaped_output.info()->tensor_shape().y() - (kernel_size.y() - conv_info.pad_bottom() - 1) - pad_valid.y();
    if(_has_bias)
    {
        _memory_group.manage(&_bias_output);
    }
    else if(_needs_permute)
    {
        output_to_use = &_permuted_output;
        _memory_group.manage(&_permuted_output);
    }
    _extract_output_func.configure(compile_context, &_reshaped_output, output_to_use, Coordinates(start_left, start_top), Coordinates(end_right, end_botton));
    _itransformed_output.allocator()->allocate();

    // Add bias
    if(biases != nullptr)
    {
        output_to_use = output;
        if(_needs_permute)
        {
            output_to_use = &_permuted_output;
            _memory_group.manage(&_permuted_output);
        }
        auto_init_if_empty(*output_to_use->info(), *_bias_output.info());
        _bias_add_func.configure(compile_context, &_bias_output, &_permuted_bias, output_to_use, ConvertPolicy::WRAP);
        _bias_output.allocator()->allocate();
    }

    // Permute output
    if(_needs_permute)
    {
        // Configure the function to transform the convoluted output to ACL's native ordering format NCHW
        _permuted_output.info()->set_data_layout(DataLayout::NCHW);
        _permute_output_func.configure(compile_context, &_permuted_output, output, PermutationVector(2U, 0U, 1U));

        // Allocate tensors
        _permuted_output.allocator()->allocate();
    }

    // Configure Activation Layer
    _is_activationlayer_enabled = act_info.enabled();
    if(_is_activationlayer_enabled)
    {
        _activation_layer_func.configure(compile_context, output, nullptr, act_info);
    }

    // Setup flip axis data
    _flip_axis.allocator()->allocate();
    _flip_axis.map(true);
    auto axis_data = reinterpret_cast<uint32_t *>(_flip_axis.buffer());
    axis_data[0]   = 0;
    axis_data[1]   = 1;
    _flip_axis.unmap();
}

Status CLFFTConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                       const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);

    // Get indices for the width and height
    const size_t idx_width  = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_height = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::HEIGHT);

    // Input shape, kernel size and output tile
    const Size2D kernel_size = Size2D(weights->tensor_shape()[idx_width], weights->tensor_shape()[idx_height]);

    // Strides
    const auto strides = conv_info.stride();
    ARM_COMPUTE_RETURN_ERROR_ON(strides.first != strides.second && strides.first != 1);
    ARM_COMPUTE_RETURN_ERROR_ON(kernel_size.x() != kernel_size.y());
    ARM_COMPUTE_RETURN_ERROR_ON(conv_info.pad_left() != (kernel_size.x() / 2) || conv_info.pad_right() != (kernel_size.x() / 2));
    ARM_COMPUTE_RETURN_ERROR_ON(conv_info.pad_top() != (kernel_size.y() / 2) || conv_info.pad_bottom() != (kernel_size.y() / 2));

    // Validate biases
    if(biases != nullptr)
    {
        const size_t idx_channels = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::CHANNEL);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
        ARM_COMPUTE_RETURN_ERROR_ON(input->tensor_shape()[idx_channels] != biases->tensor_shape().x());
    }

    // Checks performed when output is configured
    if((output != nullptr) && (output->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON((input->tensor_shape()[idx_height] != output->tensor_shape()[idx_height]) || (input->tensor_shape()[idx_width] != output->tensor_shape()[idx_width]));

        // Validate Activation Layer
        if(act_info.enabled())
        {
            ARM_COMPUTE_RETURN_ON_ERROR(CLActivationLayer::validate(output, nullptr, act_info));
        }
    }

    return Status{};
}

void CLFFTConvolutionLayer::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_memory_group);

    // Transform input
    if(_needs_permute)
    {
        _permute_input_func.run();
    }
    _pad_input_func.run();
    _transform_input_func.run();

    // Perform operations to frequency domain
    _prod_func.run();
    _reduce_func.run();

    // Transform output
    _itransform_output_func.run();
    _reshaped_output.allocator()->import_memory(_itransformed_output.cl_buffer());
    _extract_output_func.run();
    // Add bias
    if(_has_bias)
    {
        _bias_add_func.run();
    }
    if(_needs_permute)
    {
        _permute_output_func.run();
    }

    // Run activation layer
    if(_is_activationlayer_enabled)
    {
        _activation_layer_func.run();
    }
}

void CLFFTConvolutionLayer::prepare()
{
    if(!_is_prepared)
    {
        // Permute bias to NCHW
        if(_original_bias != nullptr)
        {
            _permuted_bias.allocator()->allocate();
            _permute_bias_func.run();
            _original_bias->mark_as_unused();
        }

        const ICLTensor *cur_weights = _original_weights;
        // Permute weights
        if(_needs_permute)
        {
            ARM_COMPUTE_ERROR_ON(!cur_weights->is_used());

            _permuted_weights.allocator()->allocate();
            _permute_weights_func.run();
            cur_weights->mark_as_unused();
            cur_weights = &_permuted_weights;
        }

        // Flip weights
        _flipped_weights.allocator()->allocate();
        _flip_weights_func.run();
        cur_weights->mark_as_unused();

        // Pad weights
        _padded_weights.allocator()->allocate();
        _pad_weights_func.run();
        _flipped_weights.mark_as_unused();
        CLScheduler::get().queue().finish();
        _flipped_weights.allocator()->free();

        // Transform weights to frequency domain
        _transformed_weights.allocator()->allocate();
        _transform_weights_func->run();
        _padded_weights.mark_as_unused();
        CLScheduler::get().queue().finish();
        // Delete object and release internal memory
        _transform_weights_func.reset();
        _padded_weights.allocator()->free();

        _is_prepared = true;
    }
}
} // namespace arm_compute
