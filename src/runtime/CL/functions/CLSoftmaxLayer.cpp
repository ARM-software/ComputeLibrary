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
    : _memory_group(std::move(memory_manager)), _max_shift_exp_sum_kernel(), _norm_kernel(), _flatten_kernel(), _reshape_kernel(), _max(), _sum(), _tmp(), _input_flat(), _output_flat(),
      _needs_flattening(false)
{
}

void CLSoftmaxLayer::configure_flatten_kernel(const ICLTensor *input, const ICLTensor *output)
{
    // Flatten the input
    const TensorShape shape_flatten = misc::shape_calculator::compute_flatten_shape(input->info());

    // Initialize the flat input
    _input_flat.allocator()->init(input->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(shape_flatten));

    // Configure the flatten_kernel
    _flatten_kernel.configure(input, &_input_flat);

    // We need to init the output tensor here. Indeed, the reshape kernel expects
    // both tensors to be already initialized
    auto_init_if_empty(*output->info(), *input->info()->clone());
}

void CLSoftmaxLayer::configure(const ICLTensor *input, ICLTensor *output, float beta)
{
    // Perform validation step
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(CLSoftmaxLayer::validate(input->info(), output->info()));

    _needs_flattening = input->info()->num_dimensions() > 2;

    // If we are dealing with a 4D tensor, we will:
    // - Flatten the input, so that we end up with a [width*height*depth] * batches 2D tensor
    // - Execute all the pipeline (reduction + normalization) on the flattened tensor
    // - Reshape the flattened output into the real output
    if(_needs_flattening)
    {
        // Add to the memory manager _input_flat
        _memory_group.manage(&_input_flat);

        // Cofigure  _flatten_kernel and _input_flat
        configure_flatten_kernel(input, output);
    }

    // We want to deal with a 2D input. Either it is the flattened version of the original input (4D case)
    // or it is the original input case (2D case)
    const ICLTensor *input_2D = (_needs_flattening ? &_input_flat : input);

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
        // Add to the memory manager _output_flat
        _memory_group.manage(&_output_flat);

        // The normalization kernel stores the result in a flat output tensor
        _norm_kernel.configure(&_tmp, &_sum, &_output_flat, beta);

        // Reshape the flat output into a the requested (4D) output
        _reshape_kernel.configure(&_output_flat, output);

        // Allocate the intermediate flat tensors
        _input_flat.allocator()->allocate();
        _output_flat.allocator()->allocate();
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

Status CLSoftmaxLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->num_dimensions() > 4, "Only up to 4 dimensions are supported");

    // Create intermediate tensor info
    DataType   tmp_data_type = is_data_type_quantized_asymmetric(input->data_type()) ? DataType::S32 : input->data_type();
    TensorInfo tensor_info_tmp(input->clone()->set_data_type(tmp_data_type).set_is_resizable(true));

    TensorShape max_sum_shape = input->tensor_shape();
    max_sum_shape.set(0, 1);
    TensorInfo tensor_info_max(input->clone()->set_tensor_shape(max_sum_shape).set_is_resizable(true));
    TensorInfo tensor_info_sum(input->clone()->set_tensor_shape(max_sum_shape).set_data_type(tmp_data_type).set_quantization_info(QuantizationInfo()).set_is_resizable(true));

    const TensorShape shape_flatten = misc::shape_calculator::compute_flatten_shape(input);
    TensorInfo        tensor_info_flat(input->clone()->set_tensor_shape(shape_flatten).set_is_resizable(true));

    if(input->num_dimensions() > 2) // needs flattening
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLFlattenLayerKernel::validate(input, &tensor_info_flat));
    }

    ARM_COMPUTE_RETURN_ON_ERROR(CLLogits1DMaxShiftExpSumKernel::validate(input, &tensor_info_max, &tensor_info_tmp, &tensor_info_sum));
    ARM_COMPUTE_RETURN_ON_ERROR(CLLogits1DNormKernel::validate(&tensor_info_tmp, &tensor_info_sum, output));

    return Status{};
}

void CLSoftmaxLayer::run()
{
    _memory_group.acquire();
    if(_needs_flattening)
    {
        CLScheduler::get().enqueue(_flatten_kernel, false);
    }

    CLScheduler::get().enqueue(_max_shift_exp_sum_kernel, false);
    CLScheduler::get().enqueue(_norm_kernel, !_needs_flattening);

    if(_needs_flattening)
    {
        CLScheduler::get().enqueue(_reshape_kernel, true);
    }

    // Relase intermediate buffers
    _memory_group.release();
}

} // namespace arm_compute
