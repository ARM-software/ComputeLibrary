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
#include "arm_compute/runtime/CL/functions/CLSoftmaxLayer.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/CL/kernels/CLSoftmaxLayerKernel.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

namespace arm_compute
{
CLSoftmaxLayer::CLSoftmaxLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _max_shift_exp_sum_kernel(), _norm_kernel(), _flatten_kernel_ptr(), _reshape_kernel(), _max(), _sum(), _tmp(), _input_flattened(), _output_flattened(),
      _needs_flattening(false)
{
}

void CLSoftmaxLayer::configure_reshape_input_kernel(const ICLTensor *input, const ICLTensor *output, size_t axis)
{
    // Flatten the input
    const TensorShape shape_flatten = misc::shape_calculator::compute_softmax_shape(input->info(), axis);

    // Initialize the flat input
    _input_flattened.allocator()->init(input->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(shape_flatten));

    // If we need to flatten the input, we can use CLFlattenKernel or CLReshapeKernel
    // If flattening on the third axes, we use CLFlattenKernel.
    // In all other cases we have to use CLReshapeKernel
    if(axis != 3)
    {
        auto reshape_kernel_ptr = support::cpp14::make_unique<CLReshapeLayerKernel>();
        reshape_kernel_ptr->configure(input, &_input_flattened);
        _flatten_kernel_ptr = std::move(reshape_kernel_ptr);
    }
    else
    {
        auto flatten_kernel_ptr = support::cpp14::make_unique<CLFlattenLayerKernel>();
        flatten_kernel_ptr->configure(input, &_input_flattened);
        _flatten_kernel_ptr = std::move(flatten_kernel_ptr);
    }

    // We need to init the output tensor here. Indeed, the reshape kernel expects
    // both tensors to be already initialized
    auto_init_if_empty(*output->info(), *input->info()->clone());
}

void CLSoftmaxLayer::configure(const ICLTensor *input, ICLTensor *output, float beta, size_t axis)
{
    // Perform validation step
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(CLSoftmaxLayer::validate(input->info(), output->info(), beta, axis));

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

        // Cofigure  _flatten_kernel and _input_flattened
        configure_reshape_input_kernel(input, output, axis);
    }

    // We want to deal with a 2D input. Either it is the flattened version of the original input (4D case)
    // or it is the original input case (2D case)
    const ICLTensor *input_2D = (_needs_flattening ? &_input_flattened : input);

    // Create intermediate tensors shapes
    TensorInfo input_info    = input_2D->info()->clone()->reset_padding().set_is_resizable(true);
    DataType   tmp_data_type = is_data_type_quantized_asymmetric(input_2D->info()->data_type()) ? DataType::S32 : input_2D->info()->data_type();
    TensorInfo tensor_info_tmp(input_info.clone()->set_data_type(tmp_data_type));
    _tmp.allocator()->init(tensor_info_tmp);

    TensorShape max_sum_shape = input_2D->info()->tensor_shape();
    max_sum_shape.set(0, 1);
    _max.allocator()->init(input_info.clone()->set_tensor_shape(max_sum_shape));
    _sum.allocator()->init(input_info.clone()->set_tensor_shape(max_sum_shape).set_data_type(tmp_data_type));

    // Set GPU target to kernels
    _max_shift_exp_sum_kernel.set_target(CLScheduler::get().target());

    // Manage intermediate buffers
    _memory_group.manage(&_tmp);
    _memory_group.manage(&_max);
    _memory_group.manage(&_sum);

    // Configure kernels
    _max_shift_exp_sum_kernel.configure(input_2D, &_max, &_tmp, &_sum, beta);

    if(_needs_flattening)
    {
        // Add to the memory manager _output_flattened
        _memory_group.manage(&_output_flattened);

        // The normalization kernel stores the result in a flat output tensor
        _norm_kernel.configure(&_tmp, &_sum, &_output_flattened, beta);

        // Reshape the flat output into a the requested (4D) output
        _reshape_kernel.configure(&_output_flattened, output);

        // Allocate the intermediate flat tensors
        _input_flattened.allocator()->allocate();
        _output_flattened.allocator()->allocate();
    }
    else
    {
        // Softmax 2D case
        _norm_kernel.configure(&_tmp, &_sum, output, beta);
    }

    // Allocate intermediate buffers
    _tmp.allocator()->allocate();
    _max.allocator()->allocate();
    _sum.allocator()->allocate();
}

Status CLSoftmaxLayer::validate(const ITensorInfo *input, const ITensorInfo *output, float beta, size_t axis)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->num_dimensions() > 4, "Only up to 4 dimensions are supported");
    ARM_COMPUTE_UNUSED(beta);

    // Create intermediate tensor info
    DataType   tmp_data_type = is_data_type_quantized_asymmetric(input->data_type()) ? DataType::S32 : input->data_type();
    TensorInfo tensor_info_tmp(input->clone()->set_data_type(tmp_data_type).set_is_resizable(true));

    TensorShape max_sum_shape = input->tensor_shape();
    max_sum_shape.set(0, 1);
    TensorInfo tensor_info_max(input->clone()->set_tensor_shape(max_sum_shape).set_is_resizable(true));
    TensorInfo tensor_info_sum(input->clone()->set_tensor_shape(max_sum_shape).set_data_type(tmp_data_type).set_quantization_info(QuantizationInfo()).set_is_resizable(true));

    const bool needs_flattening = (axis != 1);

    if(needs_flattening)
    {
        const TensorShape shape_flatten = misc::shape_calculator::compute_softmax_shape(input, axis);
        TensorInfo        tensor_info_flat(input->clone()->set_tensor_shape(shape_flatten).set_is_resizable(true));

        if(axis != 3)
        {
            ARM_COMPUTE_RETURN_ON_ERROR(CLReshapeLayerKernel::validate(input, &tensor_info_flat));
        }
        else
        {
            ARM_COMPUTE_RETURN_ON_ERROR(CLFlattenLayerKernel::validate(input, &tensor_info_flat));
        }
    }

    ARM_COMPUTE_RETURN_ON_ERROR(CLLogits1DMaxShiftExpSumKernel::validate(input, &tensor_info_max, &tensor_info_tmp, &tensor_info_sum));
    ARM_COMPUTE_RETURN_ON_ERROR(CLLogits1DNormKernel::validate(&tensor_info_tmp, &tensor_info_sum, output));

    if(needs_flattening)
    {
        const TensorShape shape_flatten = misc::shape_calculator::compute_softmax_shape(input);
        TensorInfo        tensor_info_flat(input->clone()->set_tensor_shape(shape_flatten).set_is_resizable(true));
    }

    return Status{};
}

void CLSoftmaxLayer::run()
{
    MemoryGroupResourceScope scope_mg(_memory_group);

    if(_needs_flattening)
    {
        CLScheduler::get().enqueue(*_flatten_kernel_ptr, false);
    }

    CLScheduler::get().enqueue(_max_shift_exp_sum_kernel, false);
    CLScheduler::get().enqueue(_norm_kernel, !_needs_flattening);

    if(_needs_flattening)
    {
        CLScheduler::get().enqueue(_reshape_kernel, true);
    }
}

} // namespace arm_compute
