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
#include "arm_compute/runtime/NEON/functions/NESoftmaxLayer.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/NEON/kernels/NESoftmaxLayerKernel.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "utils/TypePrinter.h"

#include <cfloat>

namespace arm_compute
{
NESoftmaxLayer::NESoftmaxLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _max_kernel(), _softmax_kernel(), _flat_or_reshape_kernel_ptr(nullptr), _fill_border_kernel(), _reshape_kernel(), _max(), _tmp(), _input_flattened(),
      _output_flattened(), _needs_flattening(false)
{
}

void NESoftmaxLayer::configure_reshape_input_kernel(const ITensor *input, const ITensor *output, size_t axis)
{
    // Flatten the input
    const TensorShape shape_flatten = misc::shape_calculator::compute_softmax_shape(input->info(), axis);

    // Initialize the flat input
    _input_flattened.allocator()->init(input->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(shape_flatten));

    // If we need to flatten the input, we can use NEFlattenKernel or NEReshapeKernel
    // If flattening on the third axes, we use NEFlattenKernel.
    // In all other cases we have to use NEReshapeKernel
    if(axis != 3)
    {
        auto reshape_kernel_ptr = support::cpp14::make_unique<NEReshapeLayerKernel>();
        reshape_kernel_ptr->configure(input, &_input_flattened);
        _flat_or_reshape_kernel_ptr = std::move(reshape_kernel_ptr);
    }
    else
    {
        auto flatten_kernel_ptr = support::cpp14::make_unique<NEFlattenLayerKernel>();
        flatten_kernel_ptr->configure(input, &_input_flattened);
        _flat_or_reshape_kernel_ptr = std::move(flatten_kernel_ptr);
    }

    // We need to init the output tensor here. Indeed, the reshape kernel expects
    // both tensors to be already initialized
    auto_init_if_empty(*output->info(), *input->info()->clone());
}

void NESoftmaxLayer::configure(ITensor *input, ITensor *output, float beta, size_t axis)
{
    // Perform validation step
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(NESoftmaxLayer::validate(input->info(), output->info(), beta, axis));

    // We don't need flattening only in the case the input is 2D and axis is 1
    _needs_flattening = axis != 1;

    // If we are dealing with a 4D tensor, we will:
    // - Flatten the input, so that we end up with a [width*height*depth] * batches 2D tensor
    // - Execute all the pipeline (reduction + normalization) on the flattened tensor
    // - Reshape the flattened output into the real output
    if(_needs_flattening)
    {
        // Add to the memory manager _input_flattened
        _memory_group.manage(&_input_flattened);

        // Configure  _flatten_kernel and _input_flattened
        configure_reshape_input_kernel(input, output, axis);
    }

    // We want to deal with a 2D input. Either it is the flattened version of the original input (4D case)
    // or it is the original input case (2D case)
    ITensor *input_2D = (_needs_flattening ? &_input_flattened : input);

    // Create intermediate tensors shapes
    const TensorInfo input_info    = input_2D->info()->clone()->reset_padding().set_is_resizable(true);
    DataType         tmp_data_type = is_data_type_quantized_asymmetric(input_2D->info()->data_type()) ? DataType::F32 : input_2D->info()->data_type();
    TensorInfo       tensor_info_tmp(input_info.clone()->set_data_type(tmp_data_type));

    // Init intermediate tensors
    TensorShape max_sum_shape = input_2D->info()->tensor_shape();
    max_sum_shape.set(0, 1);
    _max.allocator()->init(input_info.clone()->set_tensor_shape(max_sum_shape));
    _tmp.allocator()->init(tensor_info_tmp);

    // Manage intermediate buffers
    _memory_group.manage(&_max);
    _memory_group.manage(&_tmp);

    // Configure Kernels
    _max_kernel.configure(input_2D, &_max);
    if(_needs_flattening)
    {
        // Add to the memory manager _output_flattened
        _memory_group.manage(&_output_flattened);

        // The normalization kernel stores the result in a flat output tensor
        _softmax_kernel.configure(input_2D, &_max, &_output_flattened, beta, &_tmp);
        _input_flattened.allocator()->allocate();

        // Reshape the flat output into the requested (4D) output
        _reshape_kernel.configure(&_output_flattened, output);

        // Allocate the intermediate flat tensors
        _output_flattened.allocator()->allocate();
    }
    else
    {
        // Softmax 2D case
        _fill_border_kernel.configure(input_2D, _max_kernel.border_size(), BorderMode::REPLICATE);
        _softmax_kernel.configure(input_2D, &_max, output, beta, &_tmp);
    }

    // Allocate intermediate buffers
    _max.allocator()->allocate();
    _tmp.allocator()->allocate();
}

Status NESoftmaxLayer::validate(const ITensorInfo *input, const ITensorInfo *output, float beta, size_t axis)
{
    // Perform validation step
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->num_dimensions() > 4, "Only up to 4 dimensions are supported");
    ARM_COMPUTE_UNUSED(beta);
    ARM_COMPUTE_RETURN_ERROR_ON(axis < 1 || input->num_dimensions() < axis);

    // Create intermediate tensor info
    DataType         tmp_data_type = input->data_type();
    const TensorInfo tensor_info_tmp(input->clone()->set_data_type(tmp_data_type).set_is_resizable(true));

    TensorShape max_sum_shape = input->tensor_shape();
    max_sum_shape.set(0, 1);
    const TensorInfo tensor_info_max_sum(input->clone()->set_tensor_shape(max_sum_shape).set_data_type(tmp_data_type).set_quantization_info(input->quantization_info()).set_is_resizable(true));
    const TensorInfo dont_care;

    const bool needs_flattening = (axis != 1);

    if(needs_flattening)
    {
        const TensorShape shape_flatten = misc::shape_calculator::compute_softmax_shape(input, axis);
        TensorInfo        tensor_info_flat(input->clone()->set_tensor_shape(shape_flatten).set_is_resizable(true));

        if(axis != 3)
        {
            ARM_COMPUTE_RETURN_ON_ERROR(NEReshapeLayerKernel::validate(input, &tensor_info_flat));
        }
        else
        {
            ARM_COMPUTE_RETURN_ON_ERROR(NEFlattenLayerKernel::validate(input, &tensor_info_flat));
        }
    }

    ARM_COMPUTE_RETURN_ON_ERROR(NELogits1DMaxKernel::validate(input, &tensor_info_max_sum));
    ARM_COMPUTE_RETURN_ON_ERROR(NELogits1DSoftmaxKernel::validate(&tensor_info_tmp, &tensor_info_max_sum, output, beta, &dont_care));

    return Status{};
}

void NESoftmaxLayer::run()
{
    MemoryGroupResourceScope scope_mg(_memory_group);

    if(_needs_flattening)
    {
        NEScheduler::get().schedule(_flat_or_reshape_kernel_ptr.get(), Window::DimY);
    }

    NEScheduler::get().schedule(&_fill_border_kernel, Window::DimY);
    NEScheduler::get().schedule(&_max_kernel, Window::DimY);
    NEScheduler::get().schedule(&_softmax_kernel, Window::DimY);

    if(_needs_flattening)
    {
        NEScheduler::get().schedule(&_reshape_kernel, Window::DimY);
    }
}
} // namespace arm_compute