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
#include "src/gpu/cl/operators/ClSoftmax.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/core/helpers/SoftmaxHelpers.h"
#include "src/gpu/cl/kernels/ClSoftmaxKernel.h"
#include "src/gpu/cl/operators/ClPermute.h"
#include "src/gpu/cl/utils/ClAuxTensorHandler.h"
#include "support/Cast.h"

using namespace arm_compute::experimental;

namespace arm_compute
{
namespace opencl
{
ClSoftmax::ClSoftmax()
    : _permute_input(std::make_unique<ClPermute>()),
      _permute_output(std::make_unique<ClPermute>()),
      _max_shift_exp_sum_kernel(std::make_unique<kernels::ClLogits1DMaxShiftExpSumKernel>()),
      _norm_kernel(std::make_unique<kernels::ClLogits1DNormKernel>()),
      _max_info(),
      _sum_info(),
      _tmp_info(),
      _permuted_src_info(),
      _permuted_dst_info(),
      _aux_mem(InternalTensorIdx::COUNT)
{
}

void ClSoftmax::configure(const CLCompileContext &compile_context, const ITensorInfo &src, ITensorInfo &dst, const SoftmaxKernelInfo &info)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate(src, dst, info));

    const size_t actual_axis = static_cast<size_t>(wrap_around(info.axis, static_cast<int32_t>(src.num_dimensions())));

    _needs_permute = actual_axis != 0;

    const ITensorInfo &tmp_input_info  = _needs_permute ? _permuted_src_info : src;
    ITensorInfo       &tmp_output_info = _needs_permute ? _permuted_dst_info : dst;

    if(_needs_permute)
    {
        const auto perm_info = softmax_helpers::get_permutation_vector_from_softmax_axis(actual_axis);
        _permute_input->configure(compile_context, &src, &_permuted_src_info, perm_info);
    }

    DataType tmp_data_type = is_data_type_quantized_asymmetric(tmp_input_info.data_type()) ? DataType::S32 : tmp_input_info.data_type();
    _tmp_info              = tmp_input_info.clone()->set_data_type(tmp_data_type);

    TensorShape max_sum_shape = tmp_input_info.tensor_shape();
    _max_info                 = tmp_input_info.clone()->set_tensor_shape(max_sum_shape);
    _sum_info                 = tmp_input_info.clone()->set_tensor_shape(max_sum_shape).set_data_type(tmp_data_type);

    // Set GPU target to kernels
    _max_shift_exp_sum_kernel->set_target(CLScheduler::get().target());

    _max_shift_exp_sum_kernel->configure(compile_context, tmp_input_info, _max_info, _tmp_info, _sum_info, info);
    _norm_kernel->configure(compile_context, _tmp_info, _sum_info, tmp_output_info, info);

    if(_needs_permute)
    {
        const auto perm_info = softmax_helpers::get_permutation_vector_from_softmax_axis(actual_axis);
        _permute_output->configure(compile_context, &_permuted_dst_info, &dst, perm_info);
    }

    _aux_mem[InternalTensorIdx::SUM] = MemoryInfo(offset_int_vec(InternalTensorIdx::SUM), MemoryLifetime::Temporary, _sum_info.total_size());
    _aux_mem[InternalTensorIdx::TMP] = MemoryInfo(offset_int_vec(InternalTensorIdx::TMP), MemoryLifetime::Temporary, _tmp_info.total_size());
    _aux_mem[InternalTensorIdx::MAX] = MemoryInfo(offset_int_vec(InternalTensorIdx::MAX), MemoryLifetime::Temporary, _max_info.total_size());

    _aux_mem[InternalTensorIdx::PERMUTED_SRC] = MemoryInfo(offset_int_vec(InternalTensorIdx::PERMUTED_SRC), MemoryLifetime::Temporary, _permuted_src_info.total_size());
    _aux_mem[InternalTensorIdx::PERMUTED_DST] = MemoryInfo(offset_int_vec(InternalTensorIdx::PERMUTED_DST), MemoryLifetime::Temporary, _permuted_dst_info.total_size());
}

Status ClSoftmax::validate(const ITensorInfo &src, const ITensorInfo &dst, const SoftmaxKernelInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(src.num_dimensions() > 4, "Only up to 4 dimensions are supported");
    ARM_COMPUTE_UNUSED(info.beta);
    ARM_COMPUTE_RETURN_ERROR_ON(info.axis < static_cast<int32_t>(-src.num_dimensions()) || static_cast<int32_t>(src.num_dimensions()) <= info.axis);

    const size_t actual_axis   = static_cast<size_t>(wrap_around(info.axis, static_cast<int32_t>(src.num_dimensions())));
    const bool   needs_permute = actual_axis != 0;
    if(needs_permute)
    {
        const PermutationVector permutation_vector = softmax_helpers::get_permutation_vector_from_softmax_axis(actual_axis);
        const TensorShape       permuted_shape     = misc::shape_calculator::compute_permutation_output_shape(src, permutation_vector);
        TensorInfo              input_permuted(src.clone()->set_tensor_shape(permuted_shape));
        ARM_COMPUTE_RETURN_ON_ERROR(ClPermute::validate(&src, &input_permuted, permutation_vector));
        TensorInfo output_permuted(dst.clone()->set_tensor_shape(permuted_shape));
        ARM_COMPUTE_RETURN_ON_ERROR(ClPermute::validate(&output_permuted, &dst, permutation_vector));
    }

    // Create intermediate tensor info
    DataType   tmp_data_type = is_data_type_quantized_asymmetric(src.data_type()) ? DataType::S32 : src.data_type();
    TensorInfo tensor_info_tmp(src.clone()->set_data_type(tmp_data_type).set_is_resizable(true));

    TensorShape max_sum_shape = src.tensor_shape();
    max_sum_shape.set(0, 1);
    TensorInfo tensor_info_max(src.clone()->set_tensor_shape(max_sum_shape).set_is_resizable(true));
    TensorInfo tensor_info_sum(src.clone()->set_tensor_shape(max_sum_shape).set_data_type(tmp_data_type).set_quantization_info(QuantizationInfo()).set_is_resizable(true));

    ARM_COMPUTE_RETURN_ON_ERROR(kernels::ClLogits1DMaxShiftExpSumKernel::validate(src, tensor_info_max, tensor_info_tmp, tensor_info_sum));
    ARM_COMPUTE_RETURN_ON_ERROR(kernels::ClLogits1DNormKernel::validate(tensor_info_tmp, tensor_info_sum, dst, info));

    return Status{};
}

void ClSoftmax::run(ITensorPack &tensors)
{
    auto src = tensors.get_const_tensor(TensorType::ACL_SRC);
    auto dst = tensors.get_tensor(TensorType::ACL_DST);

    CLAuxTensorHandler sum(offset_int_vec(InternalTensorIdx::SUM), _sum_info, tensors, false);
    CLAuxTensorHandler tmp(offset_int_vec(InternalTensorIdx::TMP), _tmp_info, tensors, false);
    CLAuxTensorHandler max(offset_int_vec(InternalTensorIdx::MAX), _max_info, tensors, false);

    CLAuxTensorHandler permuted_src(offset_int_vec(InternalTensorIdx::PERMUTED_SRC), _permuted_src_info, tensors, false);
    CLAuxTensorHandler permuted_dst(offset_int_vec(InternalTensorIdx::PERMUTED_DST), _permuted_dst_info, tensors, false);

    if(_needs_permute)
    {
        ITensorPack pack;
        pack.add_const_tensor(TensorType::ACL_SRC, src);
        pack.add_tensor(TensorType::ACL_DST, permuted_src.get());
        _permute_input.get()->run(pack);
    }

    ITensorPack sum_pack;
    ITensorPack norm_pack;
    if(_needs_permute)
    {
        sum_pack.add_const_tensor(TensorType::ACL_SRC, permuted_src.get());
        norm_pack.add_tensor(TensorType::ACL_DST, permuted_dst.get());
    }
    else
    {
        sum_pack.add_const_tensor(TensorType::ACL_SRC, src);
        norm_pack.add_tensor(TensorType::ACL_DST, dst);
    }
    sum_pack.add_tensor(TensorType::ACL_DST, tmp.get());
    sum_pack.add_tensor(TensorType::ACL_INT_0, max.get());
    sum_pack.add_tensor(TensorType::ACL_INT_1, sum.get());

    norm_pack.add_const_tensor(TensorType::ACL_SRC, tmp.get());
    norm_pack.add_tensor(TensorType::ACL_INT_0, sum.get());

    CLScheduler::get().enqueue_op(*_max_shift_exp_sum_kernel.get(), sum_pack, false);
    CLScheduler::get().enqueue_op(*_norm_kernel.get(), norm_pack, false);

    if(_needs_permute)
    {
        ITensorPack pack;
        pack.add_const_tensor(TensorType::ACL_SRC, permuted_dst.get());
        pack.add_tensor(TensorType::ACL_DST, dst);
        _permute_output.get()->run(pack);
    }
}

experimental::MemoryRequirements ClSoftmax::workspace() const
{
    return _aux_mem;
}
} // namespace opencl
} // namespace arm_compute