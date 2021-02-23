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
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "src/core/CL/ICLKernel.h"
#include "src/core/CL/kernels/CLFillBorderKernel.h"
#include "src/core/CL/kernels/CLSoftmaxLayerKernel.h"
#include "src/core/helpers/SoftmaxHelpers.h"

namespace arm_compute
{
template <bool IS_LOG>
CLSoftmaxLayerGeneric<IS_LOG>::CLSoftmaxLayerGeneric(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)),
      _permute_input(),
      _permute_output(),
      _max_shift_exp_sum_kernel(std::make_unique<CLLogits1DMaxShiftExpSumKernel>()),
      _norm_kernel(std::make_unique<CLLogits1DNormKernel>()),
      _max(),
      _sum(),
      _tmp(),
      _input_permuted(),
      _output_permuted(),
      _needs_permute()
{
}

template <bool IS_LOG>
CLSoftmaxLayerGeneric<IS_LOG>::~CLSoftmaxLayerGeneric() = default;

template <bool IS_LOG>
void CLSoftmaxLayerGeneric<IS_LOG>::configure(const ICLTensor *input, ICLTensor *output, float beta, int32_t axis)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, beta, axis);
}

template <bool IS_LOG>
void CLSoftmaxLayerGeneric<IS_LOG>::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, float beta, int32_t axis)
{
    // Perform validation step
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(CLSoftmaxLayerGeneric<IS_LOG>::validate(input->info(), output->info(), beta, axis));

    const size_t actual_axis = static_cast<size_t>(wrap_around(axis, static_cast<int32_t>(input->info()->num_dimensions())));

    _needs_permute              = actual_axis != 0;
    ICLTensor       *tmp_output = output;
    const ICLTensor *tmp_input  = _needs_permute ? &_input_permuted : input;
    if(_needs_permute)
    {
        _memory_group.manage(&_input_permuted);
        _memory_group.manage(&_output_permuted);
        _permute_input.configure(compile_context, input, &_input_permuted, softmax_helpers::get_permutation_vector_from_softmax_axis(actual_axis));
        tmp_output = &_output_permuted;
    }

    // Create intermediate tensors
    DataType   tmp_data_type = is_data_type_quantized_asymmetric(tmp_input->info()->data_type()) ? DataType::S32 : tmp_input->info()->data_type();
    TensorInfo tensor_info_tmp(tmp_input->info()->clone()->set_data_type(tmp_data_type));
    _tmp.allocator()->init(tensor_info_tmp);
    TensorShape max_sum_shape = tmp_input->info()->tensor_shape();
    max_sum_shape.set(0, 1);
    _max.allocator()->init(tmp_input->info()->clone()->set_tensor_shape(max_sum_shape));
    _sum.allocator()->init(tmp_input->info()->clone()->set_tensor_shape(max_sum_shape).set_data_type(tmp_data_type));

    // Set GPU target to kernels
    _max_shift_exp_sum_kernel->set_target(CLScheduler::get().target());

    // Manage intermediate buffers
    _memory_group.manage(&_tmp);
    _memory_group.manage(&_max);
    _memory_group.manage(&_sum);

    SoftmaxKernelInfo softmax_info;
    softmax_info.beta            = beta;
    softmax_info.is_log          = IS_LOG;
    softmax_info.input_data_type = tmp_input->info()->data_type();

    // Configure kernels
    _max_shift_exp_sum_kernel->configure(compile_context, tmp_input, &_max, &_tmp, &_sum, softmax_info);
    _norm_kernel->configure(compile_context, &_tmp, &_sum, tmp_output, softmax_info);

    // Allocate intermediate buffers
    _tmp.allocator()->allocate();
    _max.allocator()->allocate();
    _sum.allocator()->allocate();
    if(_needs_permute)
    {
        _permute_output.configure(compile_context, &_output_permuted, output, softmax_helpers::get_permutation_vector_from_softmax_axis(actual_axis));
        _input_permuted.allocator()->allocate();
        _output_permuted.allocator()->allocate();
    }
}

template <bool IS_LOG>
Status CLSoftmaxLayerGeneric<IS_LOG>::validate(const ITensorInfo *input, const ITensorInfo *output, float beta, int32_t axis)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->num_dimensions() > 4, "Only up to 4 dimensions are supported");
    ARM_COMPUTE_UNUSED(beta);
    ARM_COMPUTE_RETURN_ERROR_ON(axis < static_cast<int32_t>(-input->num_dimensions()) || static_cast<int32_t>(input->num_dimensions()) <= axis);

    const size_t actual_axis   = static_cast<size_t>(wrap_around(axis, static_cast<int32_t>(input->num_dimensions())));
    const bool   needs_permute = actual_axis != 0;
    if(needs_permute)
    {
        const PermutationVector permutation_vector = softmax_helpers::get_permutation_vector_from_softmax_axis(actual_axis);
        const TensorShape       permuted_shape     = misc::shape_calculator::compute_permutation_output_shape(*input, permutation_vector);
        TensorInfo              input_permuted(input->clone()->set_tensor_shape(permuted_shape));
        ARM_COMPUTE_RETURN_ON_ERROR(CLPermute::validate(input, &input_permuted, permutation_vector));
        TensorInfo output_permuted(output->clone()->set_tensor_shape(permuted_shape));
        ARM_COMPUTE_RETURN_ON_ERROR(CLPermute::validate(&output_permuted, output, permutation_vector));
    }

    // Create intermediate tensor info
    DataType   tmp_data_type = is_data_type_quantized_asymmetric(input->data_type()) ? DataType::S32 : input->data_type();
    TensorInfo tensor_info_tmp(input->clone()->set_data_type(tmp_data_type).set_is_resizable(true));

    TensorShape max_sum_shape = input->tensor_shape();
    max_sum_shape.set(0, 1);
    TensorInfo tensor_info_max(input->clone()->set_tensor_shape(max_sum_shape).set_is_resizable(true));
    TensorInfo tensor_info_sum(input->clone()->set_tensor_shape(max_sum_shape).set_data_type(tmp_data_type).set_quantization_info(QuantizationInfo()).set_is_resizable(true));

    SoftmaxKernelInfo softmax_info;
    softmax_info.beta            = beta;
    softmax_info.is_log          = IS_LOG;
    softmax_info.input_data_type = input->data_type();

    ARM_COMPUTE_RETURN_ON_ERROR(CLLogits1DMaxShiftExpSumKernel::validate(input, &tensor_info_max, &tensor_info_tmp, &tensor_info_sum));
    ARM_COMPUTE_RETURN_ON_ERROR(CLLogits1DNormKernel::validate(&tensor_info_tmp, &tensor_info_sum, output, softmax_info));

    return Status{};
}

template <bool IS_LOG>
void           CLSoftmaxLayerGeneric<IS_LOG>::run()
{
    MemoryGroupResourceScope scope_mg(_memory_group);

    if(_needs_permute)
    {
        _permute_input.run();
    }

    CLScheduler::get().enqueue(*_max_shift_exp_sum_kernel, false);
    CLScheduler::get().enqueue(*_norm_kernel, !_needs_permute);

    if(_needs_permute)
    {
        _permute_output.run();
    }
}

template class CLSoftmaxLayerGeneric<false>;
template class CLSoftmaxLayerGeneric<true>;

} // namespace arm_compute
