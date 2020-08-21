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
#include "arm_compute/runtime/NEON/functions/NESoftmaxLayer.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/NEON/kernels/NESoftmaxLayerKernel.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

namespace arm_compute
{
template <bool IS_LOG>
NESoftmaxLayerGeneric<IS_LOG>::NESoftmaxLayerGeneric(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _permute_input(), _permute_output(), _max_kernel(), _softmax_kernel(), _fill_border_kernel(), _max(), _tmp(), _input_permuted(), _output_permuted(),
      _needs_permute(false)
{
}

template <bool IS_LOG>
void NESoftmaxLayerGeneric<IS_LOG>::configure(ITensor *input, ITensor *output, float beta, int32_t axis)
{
    // Perform validation step
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(NESoftmaxLayerGeneric::validate(input->info(), output->info(), beta, axis));

    const unsigned int actual_axis = static_cast<unsigned int>(wrap_around(axis, static_cast<int32_t>(input->info()->num_dimensions())));

    _needs_permute = actual_axis > 0;

    if(_needs_permute)
    {
        // Add to the memory manager _input_permuted
        _memory_group.manage(&_input_permuted);

        _permute_input.configure(input, &_input_permuted, get_permutation_vector_from_softmax_axis(actual_axis));
    }

    // We want to deal with a 2D input. Either it is the permuted version of the original input (4D case)
    // or it is the original input case (2D case)
    ITensor *tmp_input = (_needs_permute ? &_input_permuted : input);

    // Create intermediate tensors shapes
    const TensorInfo input_info    = tmp_input->info()->clone()->reset_padding().set_is_resizable(true);
    DataType         tmp_data_type = is_data_type_quantized_asymmetric(tmp_input->info()->data_type()) ? DataType::F32 : tmp_input->info()->data_type();
    TensorInfo       tensor_info_tmp(input_info.clone()->set_data_type(tmp_data_type));

    // Init intermediate tensors
    TensorShape max_sum_shape = tmp_input->info()->tensor_shape();
    max_sum_shape.set(0, 1);
    _max.allocator()->init(input_info.clone()->set_tensor_shape(max_sum_shape));
    _tmp.allocator()->init(tensor_info_tmp);

    // Manage intermediate buffers
    _memory_group.manage(&_max);
    _memory_group.manage(&_tmp);

    // Configure Kernels
    _max_kernel.configure(tmp_input, &_max);
    if(_needs_permute)
    {
        // Add to the memory manager _output_permuted
        _memory_group.manage(&_output_permuted);

        // The normalization kernel stores the result in a permuted output tensor
        _softmax_kernel.configure(tmp_input, &_max, &_output_permuted, beta, &_tmp);
        _input_permuted.allocator()->allocate();

        // Re-permute the permuted output into the requested (4D) output
        _permute_output.configure(&_output_permuted, output, get_permutation_vector_from_softmax_axis(actual_axis));

        // Allocate the intermediate permuted tensors
        _output_permuted.allocator()->allocate();
    }
    else
    {
        // Softmax 2D case
        _fill_border_kernel.configure(tmp_input, _max_kernel.border_size(), BorderMode::REPLICATE);
        _softmax_kernel.configure(tmp_input, &_max, output, beta, &_tmp);
    }

    // Allocate intermediate buffers
    _max.allocator()->allocate();
    _tmp.allocator()->allocate();
}

template <bool IS_LOG>
Status NESoftmaxLayerGeneric<IS_LOG>::validate(const ITensorInfo *input, const ITensorInfo *output, float beta, int32_t axis)
{
    // Perform validation step
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->num_dimensions() > 4, "Only up to 4 dimensions are supported");
    ARM_COMPUTE_UNUSED(beta);
    ARM_COMPUTE_RETURN_ERROR_ON(axis < static_cast<int32_t>(-input->num_dimensions()) || static_cast<int32_t>(input->num_dimensions()) <= axis);

    // Create intermediate tensor info
    DataType         tmp_data_type = input->data_type();
    const TensorInfo tensor_info_tmp(input->clone()->set_data_type(tmp_data_type).set_is_resizable(true));

    TensorShape max_sum_shape = input->tensor_shape();
    max_sum_shape.set(0, 1);
    const TensorInfo tensor_info_max_sum(input->clone()->set_tensor_shape(max_sum_shape).set_data_type(tmp_data_type).set_quantization_info(input->quantization_info()).set_is_resizable(true));
    const TensorInfo dont_care;

    const unsigned int actual_axis = static_cast<unsigned int>(wrap_around(axis, static_cast<int32_t>(input->num_dimensions())));

    const bool needs_permute = actual_axis > 0;

    if(needs_permute)
    {
        const PermutationVector permutation_vector = get_permutation_vector_from_softmax_axis(actual_axis);
        const TensorShape       permuted_shape     = misc::shape_calculator::compute_permutation_output_shape(*input, permutation_vector);
        TensorInfo              input_permuted(input->clone()->set_tensor_shape(permuted_shape));
        ARM_COMPUTE_RETURN_ON_ERROR(NEPermute::validate(input, &input_permuted, permutation_vector));
        TensorInfo output_permuted(output->clone()->set_tensor_shape(permuted_shape));
        ARM_COMPUTE_RETURN_ON_ERROR(NEPermute::validate(&output_permuted, output, permutation_vector));
    }

    ARM_COMPUTE_RETURN_ON_ERROR(NELogits1DMaxKernel::validate(input, &tensor_info_max_sum));
    ARM_COMPUTE_RETURN_ON_ERROR(NELogits1DSoftmaxKernel<IS_LOG>::validate(&tensor_info_tmp, &tensor_info_max_sum, output, beta, &dont_care));

    return Status{};
}

template <bool IS_LOG>
void           NESoftmaxLayerGeneric<IS_LOG>::run()
{
    MemoryGroupResourceScope scope_mg(_memory_group);

    if(_needs_permute)
    {
        _permute_input.run();
    }

    NEScheduler::get().schedule(&_fill_border_kernel, Window::DimY);
    NEScheduler::get().schedule(&_max_kernel, Window::DimY);
    NEScheduler::get().schedule(&_softmax_kernel, Window::DimY);

    if(_needs_permute)
    {
        _permute_output.run();
    }
}

template class NESoftmaxLayerGeneric<false>;
template class NESoftmaxLayerGeneric<true>;

} // namespace arm_compute
