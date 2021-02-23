/*
 * Copyright (c) 2021 Arm Limited.
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
#include "src/runtime/cpu/operators/CpuSoftmax.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/core/cpu/kernels/CpuSoftmaxKernel.h"
#include "src/core/helpers/SoftmaxHelpers.h"

namespace arm_compute
{
namespace cpu
{
template <bool IS_LOG>
CpuSoftmaxGeneric<IS_LOG>::CpuSoftmaxGeneric()
    : _permute_input(), _permute_output(), _max_kernel(), _softmax_kernel(), _max(nullptr), _tmp(nullptr), _input_permuted(nullptr), _output_permuted(nullptr), _needs_permute(false)
{
}

template <bool IS_LOG>
void CpuSoftmaxGeneric<IS_LOG>::configure(const ITensorInfo *src, ITensorInfo *dst, float beta, int32_t axis)
{
    // Perform validation step
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(CpuSoftmaxGeneric::validate(src, dst, beta, axis));

    const unsigned int actual_axis = static_cast<unsigned int>(wrap_around(axis, static_cast<int32_t>(src->num_dimensions())));

    _needs_permute = actual_axis > 0;

    if(_needs_permute)
    {
        _input_permuted = std::make_unique<TensorInfo>();
        _permute_input.configure(src, _input_permuted.get(), softmax_helpers::get_permutation_vector_from_softmax_axis(actual_axis));
    }

    // We want to deal with a 2D input. Either it is the permuted version of the original input (4D case)
    // or it is the original input case (2D case)
    const ITensorInfo *tmp_input = (_needs_permute ? _input_permuted.get() : src);

    // Create intermediate tensors shapes
    TensorShape max_sum_shape = tmp_input->tensor_shape();
    max_sum_shape.set(0, 1);
    const TensorInfo input_info    = tmp_input->clone()->reset_padding().set_is_resizable(true);
    DataType         tmp_data_type = is_data_type_quantized_asymmetric(tmp_input->data_type()) ? DataType::F32 : tmp_input->data_type();
    TensorInfo       tensor_info_tmp(input_info.clone()->set_data_type(tmp_data_type));
    TensorInfo       max_info(tmp_input->clone()->set_tensor_shape(max_sum_shape));

    // Init intermediate tensors
    _max = std::make_unique<TensorInfo>(max_info);
    _tmp = std::make_unique<TensorInfo>(tensor_info_tmp);

    // Configure kernels
    auto mk = std::make_unique<kernels::CpuLogits1DMaxKernel>();
    mk->configure(tmp_input, _max.get());
    _max_kernel = std::move(mk);

    auto sm = std::make_unique<kernels::CpuLogits1DSoftmaxKernel<IS_LOG>>();
    if(_needs_permute)
    {
        _output_permuted = std::make_unique<TensorInfo>();

        // The normalization kernel stores the result in a permuted output tensor
        sm->configure(tmp_input, _max.get(), _output_permuted.get(), beta, _tmp.get());

        // Re-permute the permuted output into the requested (4D) output
        _permute_output.configure(_output_permuted.get(), dst, softmax_helpers::get_permutation_vector_from_softmax_axis(actual_axis));
    }
    else
    {
        // Softmax 2D case
        sm->configure(tmp_input, _max.get(), dst, beta, _tmp.get());
    }
    _softmax_kernel = std::move(sm);
}

template <bool IS_LOG>
Status CpuSoftmaxGeneric<IS_LOG>::validate(const ITensorInfo *src, const ITensorInfo *dst, float beta, int32_t axis)
{
    // Perform validation step
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(src->num_dimensions() > 4, "Only up to 4 dimensions are supported");
    ARM_COMPUTE_UNUSED(beta);
    ARM_COMPUTE_RETURN_ERROR_ON(axis < static_cast<int32_t>(-src->num_dimensions()) || static_cast<int32_t>(src->num_dimensions()) <= axis);

    // Create intermediate tensor info
    DataType         tmp_data_type = src->data_type();
    const TensorInfo tensor_info_tmp(src->clone()->set_data_type(tmp_data_type).set_is_resizable(true));

    TensorShape max_sum_shape = src->tensor_shape();
    max_sum_shape.set(0, 1);
    const TensorInfo tensor_info_max_sum(src->clone()->set_tensor_shape(max_sum_shape).set_data_type(tmp_data_type).set_quantization_info(src->quantization_info()).set_is_resizable(true));
    const TensorInfo dont_care;

    const unsigned int actual_axis = static_cast<unsigned int>(wrap_around(axis, static_cast<int32_t>(src->num_dimensions())));

    const bool needs_permute = actual_axis > 0;

    if(needs_permute)
    {
        const PermutationVector permutation_vector = softmax_helpers::get_permutation_vector_from_softmax_axis(actual_axis);
        const TensorShape       permuted_shape     = misc::shape_calculator::compute_permutation_output_shape(*src, permutation_vector);
        TensorInfo              input_permuted(src->clone()->set_tensor_shape(permuted_shape));
        ARM_COMPUTE_RETURN_ON_ERROR(CpuPermute::validate(src, &input_permuted, permutation_vector));
        TensorInfo output_permuted(dst->clone()->set_tensor_shape(permuted_shape));
        ARM_COMPUTE_RETURN_ON_ERROR(CpuPermute::validate(&output_permuted, dst, permutation_vector));
    }

    ARM_COMPUTE_RETURN_ON_ERROR(kernels::CpuLogits1DMaxKernel::validate(src, &tensor_info_max_sum));
    ARM_COMPUTE_RETURN_ON_ERROR(kernels::CpuLogits1DSoftmaxKernel<IS_LOG>::validate(&tensor_info_tmp, &tensor_info_max_sum, dst, beta, &dont_care));

    return Status{};
}

template <bool IS_LOG>
void CpuSoftmaxGeneric<IS_LOG>::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");

    ITensorPack max_pack;
    ITensorPack softmax_pack;

    if(_needs_permute)
    {
        ITensorPack permute_in_pack;
        permute_in_pack.add_tensor(TensorType::ACL_SRC, tensors.get_const_tensor(ACL_SRC));
        permute_in_pack.add_tensor(TensorType::ACL_DST, tensors.get_tensor(ACL_INT_2));
        _permute_input.run(permute_in_pack);

        max_pack.add_tensor(TensorType::ACL_SRC, tensors.get_tensor(ACL_INT_2));

        softmax_pack.add_tensor(TensorType::ACL_SRC_0, tensors.get_tensor(ACL_INT_2));
        softmax_pack.add_tensor(TensorType::ACL_SRC_1, tensors.get_tensor(ACL_INT_1));
        softmax_pack.add_tensor(TensorType::ACL_DST_0, tensors.get_tensor(ACL_INT_3));
        softmax_pack.add_tensor(TensorType::ACL_DST_1, tensors.get_tensor(ACL_INT_0));
    }
    else
    {
        max_pack.add_tensor(TensorType::ACL_SRC, tensors.get_const_tensor(ACL_SRC));
        softmax_pack.add_tensor(TensorType::ACL_SRC_0, tensors.get_const_tensor(ACL_SRC));
        softmax_pack.add_tensor(TensorType::ACL_SRC_1, tensors.get_tensor(ACL_INT_1));
        softmax_pack.add_tensor(TensorType::ACL_DST_0, tensors.get_tensor(ACL_DST));
        softmax_pack.add_tensor(TensorType::ACL_DST_1, tensors.get_tensor(ACL_INT_0));
    }

    max_pack.add_tensor(TensorType::ACL_DST, tensors.get_tensor(ACL_INT_1));

    NEScheduler::get().schedule_op(_max_kernel.get(), Window::DimY, _max_kernel->window(), max_pack);
    NEScheduler::get().schedule_op(_softmax_kernel.get(), Window::DimY, _softmax_kernel->window(), softmax_pack);

    if(_needs_permute)
    {
        ITensorPack permute_out_pack;
        permute_out_pack.add_tensor(TensorType::ACL_SRC, tensors.get_tensor(ACL_INT_3));
        permute_out_pack.add_tensor(TensorType::ACL_DST, tensors.get_tensor(ACL_DST));
        _permute_output.run(permute_out_pack);
    }
}

template <bool                   IS_LOG>
experimental::MemoryRequirements CpuSoftmaxGeneric<IS_LOG>::workspace() const
{
    experimental::MemoryRequirements req{};

    req.push_back({ TensorType::ACL_INT_0, _tmp->total_size(), 0 });
    req.push_back({ TensorType::ACL_INT_1, _max->total_size(), 0 });

    if(_needs_permute)
    {
        req.push_back({ TensorType::ACL_INT_2, _input_permuted->total_size(), 0 });
        req.push_back({ TensorType::ACL_INT_3, _output_permuted->total_size(), 0 });
    }

    return req;
}

template class CpuSoftmaxGeneric<false>;
template class CpuSoftmaxGeneric<true>;
} // namespace cpu
} // namespace arm_compute
