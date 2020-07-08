/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/runtime/CL/CLScheduler.h"

namespace arm_compute
{
template <bool IS_LOG>
CLSoftmaxLayerGeneric<IS_LOG>::CLSoftmaxLayerGeneric(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _max_shift_exp_sum_kernel(), _norm_kernel(), _flatten_ptr(), _reshape(), _max(), _sum(), _tmp(), _input_flattened(), _output_flattened(),
      _needs_flattening(false)
{
}

template <bool IS_LOG>
void CLSoftmaxLayerGeneric<IS_LOG>::configure_reshape_input_kernel(const ICLTensor *input, const ICLTensor *output, size_t first_n_reduce_axes)
{
    configure_reshape_input_kernel(CLKernelLibrary::get().get_compile_context(), input, output, first_n_reduce_axes);
}

template <bool IS_LOG>
void CLSoftmaxLayerGeneric<IS_LOG>::configure_reshape_input_kernel(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *output, size_t first_n_reduce_axes)
{
    // Flatten the input
    const TensorShape shape_flatten = misc::shape_calculator::compute_softmax_shape(input->info(), first_n_reduce_axes);

    // Initialize the flat input
    _input_flattened.allocator()->init(input->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(shape_flatten));

    // If we need to flatten the input, we can use CLFlattenKernel or CLReshapeKernel
    // If the number of reduced axes is 3 (max dimension), which means collapsing all axes except the batch axis, we use CLFlattenKernel.
    // In all other cases we have to use CLReshapeKernel
    // Note that the "other cases" include both:
    //   1. first_n_reduce_axes < 3: Reduce the first 1 (no need to reduce) or 2 dimensions (inclusive)
    //   2. first_n_reduce_axes == 4: Reduce all 4 dimensions. This can only be handled by CLReshapeKernel instead of CLFlattenKernel.
    if(first_n_reduce_axes == 3)
    {
        auto flatten = support::cpp14::make_unique<CLFlattenLayer>();
        flatten->configure(compile_context, input, &_input_flattened);
        _flatten_ptr = std::move(flatten);
    }
    else
    {
        auto reshape_ptr = support::cpp14::make_unique<CLReshapeLayer>();
        reshape_ptr->configure(compile_context, input, &_input_flattened);
        _flatten_ptr = std::move(reshape_ptr);
    }

    // We need to init the output tensor here. Indeed, the reshape kernel expects
    // both tensors to be already initialized
    auto_init_if_empty(*output->info(), *input->info()->clone());
}

template <bool IS_LOG>
void CLSoftmaxLayerGeneric<IS_LOG>::configure(const ICLTensor *input, ICLTensor *output, float beta, size_t reduce_end_axis)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, beta, reduce_end_axis);
}

template <bool IS_LOG>
void CLSoftmaxLayerGeneric<IS_LOG>::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, float beta, size_t reduce_end_axis)
{
    // Perform validation step
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(CLSoftmaxLayerGeneric<IS_LOG>::validate(input->info(), output->info(), beta, reduce_end_axis));

    // Convert reduce-before axis (inclusive) to first n axes to reduce
    size_t first_n_reduce_axes = dim_index_2_num_dims(reduce_end_axis, input->info()->num_dimensions());

    // We only need flattening when the number of axes to reduce is greater than 1
    _needs_flattening = first_n_reduce_axes > 1;

    // If we are dealing with a 4D tensor, we will:
    // - Flatten the input, so that we end up with a [width*height*depth] * batches 2D tensor
    // - Execute all the pipeline (reduction + normalization) on the flattened tensor
    // - Reshape the flattened output into the real output
    if(_needs_flattening)
    {
        // Add to the memory manager _input_flattened
        _memory_group.manage(&_input_flattened);

        // Cofigure _flatten_kernel and _input_flattened
        configure_reshape_input_kernel(input, output, first_n_reduce_axes);
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

    SoftmaxKernelInfo softmax_info;
    softmax_info.beta            = beta;
    softmax_info.is_log          = IS_LOG;
    softmax_info.input_data_type = input_2D->info()->data_type();

    // Configure kernels
    _max_shift_exp_sum_kernel.configure(compile_context, input_2D, &_max, &_tmp, &_sum, softmax_info);

    if(_needs_flattening)
    {
        // Add to the memory manager _output_flattened
        _memory_group.manage(&_output_flattened);

        // The normalization kernel stores the result in a flat output tensor
        _norm_kernel.configure(compile_context, &_tmp, &_sum, &_output_flattened, softmax_info);

        // Reshape the flat output into a the requested (4D) output
        _reshape.configure(compile_context, &_output_flattened, output);

        // Allocate the intermediate flat tensors
        _input_flattened.allocator()->allocate();
        _output_flattened.allocator()->allocate();
    }
    else
    {
        // Softmax 2D case
        _norm_kernel.configure(compile_context, &_tmp, &_sum, output, softmax_info);
    }

    // Allocate intermediate buffers
    _tmp.allocator()->allocate();
    _max.allocator()->allocate();
    _sum.allocator()->allocate();
}

template <bool IS_LOG>
Status CLSoftmaxLayerGeneric<IS_LOG>::validate(const ITensorInfo *input, const ITensorInfo *output, float beta, size_t reduce_end_axis)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->num_dimensions() > 4, "Only up to 4 dimensions are supported");
    ARM_COMPUTE_UNUSED(beta);
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() <= reduce_end_axis);

    // Convert reduce-before axis (inclusive) to first n axes to reduce
    size_t first_n_reduce_axes = dim_index_2_num_dims(reduce_end_axis, input->num_dimensions());

    // Create intermediate tensor info
    DataType   tmp_data_type = is_data_type_quantized_asymmetric(input->data_type()) ? DataType::S32 : input->data_type();
    TensorInfo tensor_info_tmp(input->clone()->set_data_type(tmp_data_type).set_is_resizable(true));

    TensorShape max_sum_shape = input->tensor_shape();
    max_sum_shape.set(0, 1);
    TensorInfo tensor_info_max(input->clone()->set_tensor_shape(max_sum_shape).set_is_resizable(true));
    TensorInfo tensor_info_sum(input->clone()->set_tensor_shape(max_sum_shape).set_data_type(tmp_data_type).set_quantization_info(QuantizationInfo()).set_is_resizable(true));

    const bool needs_flattening = (first_n_reduce_axes > 1);

    if(needs_flattening)
    {
        const TensorShape shape_flatten = misc::shape_calculator::compute_softmax_shape(input, first_n_reduce_axes);
        TensorInfo        tensor_info_flat(input->clone()->set_tensor_shape(shape_flatten).set_is_resizable(true));

        if(first_n_reduce_axes == 3)
        {
            ARM_COMPUTE_RETURN_ON_ERROR(CLFlattenLayer::validate(input, &tensor_info_flat));
        }
        else
        {
            ARM_COMPUTE_RETURN_ON_ERROR(CLReshapeLayer::validate(input, &tensor_info_flat));
        }
    }

    SoftmaxKernelInfo softmax_info;
    softmax_info.beta            = beta;
    softmax_info.is_log          = IS_LOG;
    softmax_info.input_data_type = input->data_type();

    ARM_COMPUTE_RETURN_ON_ERROR(CLLogits1DMaxShiftExpSumKernel::validate(input, &tensor_info_max, &tensor_info_tmp, &tensor_info_sum));
    ARM_COMPUTE_RETURN_ON_ERROR(CLLogits1DNormKernel::validate(&tensor_info_tmp, &tensor_info_sum, output, softmax_info));

    if(needs_flattening)
    {
        const TensorShape shape_flatten = misc::shape_calculator::compute_softmax_shape(input);
        TensorInfo        tensor_info_flat(input->clone()->set_tensor_shape(shape_flatten).set_is_resizable(true));
    }

    return Status{};
}

template <bool IS_LOG>
void           CLSoftmaxLayerGeneric<IS_LOG>::run()
{
    MemoryGroupResourceScope scope_mg(_memory_group);

    if(_needs_flattening)
    {
        _flatten_ptr->run();
    }

    CLScheduler::get().enqueue(_max_shift_exp_sum_kernel, false);
    CLScheduler::get().enqueue(_norm_kernel, !_needs_flattening);

    if(_needs_flattening)
    {
        _reshape.run();
    }
}

template class CLSoftmaxLayerGeneric<false>;
template class CLSoftmaxLayerGeneric<true>;

} // namespace arm_compute
