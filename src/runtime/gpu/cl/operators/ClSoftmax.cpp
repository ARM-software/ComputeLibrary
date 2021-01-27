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
#include "src/runtime/gpu/cl/operators/ClSoftmax.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/gpu/cl/kernels/ClSoftmaxKernel.h"
#include "src/core/helpers/SoftmaxHelpers.h"
#include "src/runtime/gpu/cl/operators/ClPermute.h"
#include "support/Cast.h"

namespace arm_compute
{
namespace opencl
{
namespace
{
void run_permute(ClPermute *op, const ITensor *src, ITensor *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst, op);
    ITensorPack pack;
    pack.add_const_tensor(TensorType::ACL_SRC, src);
    pack.add_tensor(TensorType::ACL_DST, dst);
    op->run(pack);
}
} // namespace

ClSoftmax::ClSoftmax()
    : _permute_input(std::make_unique<ClPermute>()),
      _permute_output(std::make_unique<ClPermute>()),
      _max_shift_exp_sum_kernel(std::make_unique<kernels::ClLogits1DMaxShiftExpSumKernel>()),
      _norm_kernel(std::make_unique<kernels::ClLogits1DNormKernel>()),
      _max_info(_internal_info[static_cast<uint32_t>(InternalTensorIdx::MAX)]),
      _sum_info(_internal_info[static_cast<uint32_t>(InternalTensorIdx::SUM)]),
      _tmp_info(_internal_info[static_cast<uint32_t>(InternalTensorIdx::TMP)]),
      _permuted_src_info(_internal_info[static_cast<uint32_t>(InternalTensorIdx::PERMUTED_SRC)]),
      _permuted_dst_info(_internal_info[static_cast<uint32_t>(InternalTensorIdx::PERMUTED_DST)])
{
}

TensorType ClSoftmax::convert_internal_idx_to_tensor_type(InternalTensorIdx idx) const
{
    switch(idx)
    {
        case InternalTensorIdx::MAX:
            return TensorType::ACL_INT_0;
        case InternalTensorIdx::SUM:
            return TensorType::ACL_INT_1;
        case InternalTensorIdx::TMP:
            return TensorType::ACL_INT_2;
        case InternalTensorIdx::PERMUTED_SRC:
            return TensorType::ACL_INT_3;
        case InternalTensorIdx::PERMUTED_DST:
            return TensorType::ACL_INT_4;
        default:
            ARM_COMPUTE_ERROR("invalid internal tensor index is given.");
            break;
    };
    return TensorType::ACL_UNKNOWN;
}

void ClSoftmax::create_internal_tensor(TensorInfo &info, InternalTensorIdx idx)
{
    const auto tensor_idx = static_cast<uint32_t>(idx);
    if(!_internal_tensor[tensor_idx])
    {
        _internal_tensor[tensor_idx] = std::make_unique<CLTensor>();
    }
    _internal_tensor[tensor_idx]->allocator()->init(info);
}

void ClSoftmax::create_internal_tensor()
{
    for(uint32_t i = 0; i < static_cast<uint32_t>(InternalTensorIdx::COUNT); i++)
    {
        const auto tensor_idx = static_cast<InternalTensorIdx>(i);

        if(!_needs_permute && (tensor_idx == InternalTensorIdx::PERMUTED_DST || tensor_idx == InternalTensorIdx::PERMUTED_SRC))
        {
            continue;
        }
        create_internal_tensor(_internal_info[i], static_cast<InternalTensorIdx>(i));
    }
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

void ClSoftmax::import_workspace_memory(ITensorPack &tensors)
{
    auto import_workspace_memory = [this, &tensors](InternalTensorIdx idx)
    {
        const auto workspace_idx   = convert_internal_idx_to_tensor_type(idx);
        auto       imported_tensor = tensors.get_tensor(workspace_idx);
        if(imported_tensor)
        {
            auto imported_memory = utils::cast::polymorphic_downcast<ICLTensor *>(imported_tensor)->cl_buffer();
            _internal_tensor[static_cast<uint32_t>(idx)].get()->allocator()->import_memory(imported_memory);
        }
    };

    import_workspace_memory(InternalTensorIdx::PERMUTED_SRC);
    import_workspace_memory(InternalTensorIdx::PERMUTED_DST);
    import_workspace_memory(InternalTensorIdx::MAX);
    import_workspace_memory(InternalTensorIdx::SUM);
    import_workspace_memory(InternalTensorIdx::TMP);
}

void ClSoftmax::run_source_permute(const ITensor *src)
{
    if(_needs_permute)
    {
        auto permuted_src = _internal_tensor[static_cast<uint32_t>(InternalTensorIdx::PERMUTED_SRC)].get();
        run_permute(_permute_input.get(), src, permuted_src);
    }
}

void ClSoftmax::run_destination_permute(ITensor *dst)
{
    if(_needs_permute)
    {
        auto permuted_dst = _internal_tensor[static_cast<uint32_t>(InternalTensorIdx::PERMUTED_DST)].get();
        run_permute(_permute_output.get(), permuted_dst, dst);
    }
}

void ClSoftmax::run_max_sum(const ITensor *src)
{
    auto max = _internal_tensor[static_cast<uint32_t>(InternalTensorIdx::MAX)].get();
    auto sum = _internal_tensor[static_cast<uint32_t>(InternalTensorIdx::SUM)].get();
    auto tmp = _internal_tensor[static_cast<uint32_t>(InternalTensorIdx::TMP)].get();

    ARM_COMPUTE_ERROR_ON_NULLPTR(src, tmp, max, sum);

    ITensorPack sum_pack;
    sum_pack.add_const_tensor(TensorType::ACL_SRC, src);
    sum_pack.add_tensor(TensorType::ACL_DST, tmp);
    sum_pack.add_tensor(TensorType::ACL_INT_0, max);
    sum_pack.add_tensor(TensorType::ACL_INT_1, sum);

    CLScheduler::get().enqueue_op(*_max_shift_exp_sum_kernel.get(), sum_pack, false);
}

void ClSoftmax::run_norm(ITensor *dst)
{
    auto sum = _internal_tensor[static_cast<uint32_t>(InternalTensorIdx::SUM)].get();
    auto tmp = _internal_tensor[static_cast<uint32_t>(InternalTensorIdx::TMP)].get();

    ARM_COMPUTE_ERROR_ON_NULLPTR(tmp, sum, dst);

    ITensorPack norm_pack;
    norm_pack.add_const_tensor(TensorType::ACL_SRC, tmp);
    norm_pack.add_tensor(TensorType::ACL_DST, dst);
    norm_pack.add_tensor(TensorType::ACL_INT_0, sum);

    CLScheduler::get().enqueue_op(*_norm_kernel.get(), norm_pack, false);
}

void ClSoftmax::run(ITensorPack &tensors)
{
    create_internal_tensor();

    auto src = tensors.get_const_tensor(TensorType::ACL_SRC);
    auto dst = tensors.get_tensor(TensorType::ACL_DST);

    import_workspace_memory(tensors);
    run_source_permute(src);
    run_max_sum(!_needs_permute ? src : _internal_tensor[static_cast<uint32_t>(InternalTensorIdx::PERMUTED_SRC)].get());
    run_norm(!_needs_permute ? dst : _internal_tensor[static_cast<uint32_t>(InternalTensorIdx::PERMUTED_DST)].get());
    run_destination_permute(dst);
}

experimental::MemoryRequirements ClSoftmax::workspace() const
{
    experimental::MemoryRequirements req{};

    req.emplace_back(convert_internal_idx_to_tensor_type(InternalTensorIdx::SUM), _sum_info.total_size(), 0);
    req.emplace_back(convert_internal_idx_to_tensor_type(InternalTensorIdx::TMP), _tmp_info.total_size(), 0);
    req.emplace_back(convert_internal_idx_to_tensor_type(InternalTensorIdx::MAX), _max_info.total_size(), 0);

    if(_needs_permute)
    {
        req.emplace_back(convert_internal_idx_to_tensor_type(InternalTensorIdx::PERMUTED_SRC), _permuted_src_info.total_size(), 0);
        req.emplace_back(convert_internal_idx_to_tensor_type(InternalTensorIdx::PERMUTED_DST), _permuted_dst_info.total_size(), 0);
    }

    return req;
}
} // namespace opencl
} // namespace arm_compute