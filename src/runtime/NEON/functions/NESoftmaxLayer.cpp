/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/Tensor.h"
#include "src/core/cpu/kernels/CpuSoftmaxKernel.h"
#include "src/core/helpers/SoftmaxHelpers.h"
#include "src/runtime/cpu/operators/CpuSoftmax.h"

namespace arm_compute
{
template <bool IS_LOG>
struct NESoftmaxLayerGeneric<IS_LOG>::Impl
{
    const ITensor                                  *src{ nullptr };
    ITensor                                        *dst{ nullptr };
    Tensor                                          max{ nullptr };
    Tensor                                          tmp{ nullptr };
    Tensor                                          input_permuted{ nullptr };
    Tensor                                          output_permuted{ nullptr };
    std::unique_ptr<cpu::CpuSoftmaxGeneric<IS_LOG>> op{ nullptr };
};

template <bool IS_LOG>
NESoftmaxLayerGeneric<IS_LOG>::NESoftmaxLayerGeneric(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _impl(std::make_unique<Impl>())
{
}

template <bool IS_LOG>
NESoftmaxLayerGeneric<IS_LOG>::NESoftmaxLayerGeneric(NESoftmaxLayerGeneric &&) = default;
template <bool                 IS_LOG>
NESoftmaxLayerGeneric<IS_LOG> &NESoftmaxLayerGeneric<IS_LOG>::operator=(NESoftmaxLayerGeneric &&) = default;
template <bool                 IS_LOG>
NESoftmaxLayerGeneric<IS_LOG>::~NESoftmaxLayerGeneric() = default;

template <bool IS_LOG>
void NESoftmaxLayerGeneric<IS_LOG>::configure(ITensor *input, ITensor *output, float beta, int32_t axis)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    _impl->src = input;
    _impl->dst = output;
    _impl->op  = std::make_unique<cpu::CpuSoftmaxGeneric<IS_LOG>>();
    _impl->op->configure(input->info(), output->info(), beta, axis);

    const unsigned int actual_axis   = static_cast<unsigned int>(wrap_around(axis, static_cast<int32_t>(input->info()->num_dimensions())));
    const bool         needs_permute = actual_axis > 0;
    if(needs_permute)
    {
        // Add to the memory manager _input_permuted
        auto permute_input = std::make_unique<cpu::CpuPermute>();
        _memory_group.manage(&_impl->input_permuted);
        permute_input->configure(input->info(), _impl->input_permuted.info(), softmax_helpers::get_permutation_vector_from_softmax_axis(actual_axis));
    }

    // We want to deal with a 2D input. Either it is the permuted version of the original input (4D case)
    // or it is the original input case (2D case)
    ITensor *tmp_input = (needs_permute ? &_impl->input_permuted : input);

    // Create intermediate tensors shapes
    const TensorInfo input_info    = tmp_input->info()->clone()->reset_padding().set_is_resizable(true);
    DataType         tmp_data_type = is_data_type_quantized_asymmetric(tmp_input->info()->data_type()) ? DataType::F32 : tmp_input->info()->data_type();
    TensorInfo       tensor_info_tmp(input_info.clone()->set_data_type(tmp_data_type));

    // Init intermediate tensors
    TensorShape max_sum_shape = tmp_input->info()->tensor_shape();
    max_sum_shape.set(0, 1);
    _impl->max.allocator()->init(input_info.clone()->set_tensor_shape(max_sum_shape));
    _impl->tmp.allocator()->init(tensor_info_tmp);

    // Manage intermediate buffers
    _memory_group.manage(&_impl->max);
    _memory_group.manage(&_impl->tmp);

    // Configure kernels
    auto max_kernel     = std::make_unique<cpu::kernels::CpuLogits1DMaxKernel>();
    auto softmax_kernel = std::make_unique<cpu::kernels::CpuLogits1DSoftmaxKernel<IS_LOG>>();
    max_kernel->configure(tmp_input->info(), _impl->max.info());

    if(needs_permute)
    {
        auto permute_output = std::make_unique<cpu::CpuPermute>();
        // Add to the memory manager _output_permuted
        _memory_group.manage(&_impl->output_permuted);

        // The normalization kernel stores the result in a permuted output tensor
        softmax_kernel->configure(tmp_input->info(), _impl->max.info(), _impl->output_permuted.info(), beta, _impl->tmp.info());
        _impl->input_permuted.allocator()->allocate();

        // Re-permute the permuted output into the requested (4D) output
        permute_output->configure(_impl->output_permuted.info(), output->info(), softmax_helpers::get_permutation_vector_from_softmax_axis(actual_axis));

        // Allocate the intermediate permuted tensors
        _impl->output_permuted.allocator()->allocate();
    }
    else
    {
        softmax_kernel->configure(tmp_input->info(), _impl->max.info(), output->info(), beta, _impl->tmp.info());
    }

    // Allocate intermediate buffers
    _impl->max.allocator()->allocate();
    _impl->tmp.allocator()->allocate();
}

template <bool IS_LOG>
Status NESoftmaxLayerGeneric<IS_LOG>::validate(const ITensorInfo *input, const ITensorInfo *output, float beta, int32_t axis)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ON_ERROR(cpu::CpuSoftmaxGeneric<IS_LOG>::validate(input, output, beta, axis));
    return Status{};
}

template <bool IS_LOG>
void           NESoftmaxLayerGeneric<IS_LOG>::run()
{
    MemoryGroupResourceScope scope_mg(_memory_group);
    ITensorPack              pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    pack.add_tensor(TensorType::ACL_INT_0, &_impl->tmp);
    pack.add_tensor(TensorType::ACL_INT_1, &_impl->max);
    pack.add_tensor(TensorType::ACL_INT_2, &_impl->input_permuted);
    pack.add_tensor(TensorType::ACL_INT_3, &_impl->output_permuted);
    _impl->op->run(pack);
}

template class NESoftmaxLayerGeneric<false>;
template class NESoftmaxLayerGeneric<true>;

} // namespace arm_compute
