/*
 * Copyright (c) 2021, 2023-2024 Arm Limited.
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
#include "src/cpu/operators/CpuSoftmax.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/common/utils/Log.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/core/helpers/SoftmaxHelpers.h"
#include "src/cpu/kernels/CpuSoftmaxKernel.h"
#include "src/cpu/utils/CpuAuxTensorHandler.h"

using namespace arm_compute::experimental;

namespace arm_compute
{
namespace cpu
{
CpuSoftmaxGeneric::CpuSoftmaxGeneric() : _softmax_kernel(), _tmp(), _aux_mem(InternalTensorIdx::COUNT)
{
}

void CpuSoftmaxGeneric::configure(const ITensorInfo *src, ITensorInfo *dst, float beta, int32_t axis, bool is_log)
{
    // Perform validation step
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(CpuSoftmaxGeneric::validate(src, dst, beta, axis));
    ARM_COMPUTE_LOG_PARAMS(src, dst, beta, axis);

    const unsigned int actual_axis =
        static_cast<unsigned int>(wrap_around(axis, static_cast<int32_t>(src->num_dimensions())));

    _axis = actual_axis;

    const ITensorInfo *tmp_input = src;

    TensorInfo tensor_info_tmp;
    if (is_data_type_quantized_asymmetric(src->data_type()))
    {
        // Create intermediate tensors shapes
        const TensorInfo input_info = tmp_input->clone()->reset_padding().set_is_resizable(true);
        tensor_info_tmp             = input_info.clone()->set_data_type(DataType::F32);
    }

    // Init intermediate tensors
    _tmp = TensorInfo(tensor_info_tmp);

    // Configure kernels
    auto sm = std::make_unique<kernels::CpuSoftmaxKernel>();

    // Softmax 2D case
    sm->configure(tmp_input, dst, beta, is_log, actual_axis, &_tmp);

    _softmax_kernel = std::move(sm);

    if (_tmp.total_size() > 0)
    {
        _aux_mem[InternalTensorIdx::TMP] =
            MemoryInfo(offset_int_vec(InternalTensorIdx::TMP), MemoryLifetime::Temporary, _tmp.total_size());
    }
}

Status
CpuSoftmaxGeneric::validate(const ITensorInfo *src, const ITensorInfo *dst, float beta, int32_t axis, bool is_log)
{
    // Perform validation step
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(src->num_dimensions() > 4, "Only up to 4 dimensions are supported");
    ARM_COMPUTE_UNUSED(beta);
    ARM_COMPUTE_RETURN_ERROR_ON(axis < static_cast<int32_t>(-src->num_dimensions()) ||
                                static_cast<int32_t>(src->num_dimensions()) <= axis);

    // Create intermediate tensor info
    TensorInfo tensor_info_tmp;

    if (is_data_type_quantized_asymmetric(src->data_type()))
    {
        tensor_info_tmp = src->clone()->set_data_type(DataType::F32).set_is_resizable(true);
    }
    const unsigned int actual_axis =
        static_cast<unsigned int>(wrap_around(axis, static_cast<int32_t>(src->num_dimensions())));

    ARM_COMPUTE_RETURN_ON_ERROR(
        kernels::CpuSoftmaxKernel::validate(src, dst, beta, actual_axis, is_log, &tensor_info_tmp));

    return Status{};
}

void CpuSoftmaxGeneric::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");

    auto src = tensors.get_const_tensor(TensorType::ACL_SRC);
    auto dst = tensors.get_tensor(TensorType::ACL_DST);

    CpuAuxTensorHandler tmp(offset_int_vec(InternalTensorIdx::TMP), _tmp, tensors, true);

    ITensorPack softmax_pack;

    softmax_pack = {{TensorType::ACL_SRC_0, src}, {TensorType::ACL_DST_0, dst}, {TensorType::ACL_DST_1, tmp.get()}};

    if (_axis == 0)
    {
        NEScheduler::get().schedule_op(_softmax_kernel.get(), Window::DimY, _softmax_kernel->window(), softmax_pack);
    }
    else
    {
        NEScheduler::get().schedule_op(_softmax_kernel.get(), Window::DimX, _softmax_kernel->window(), softmax_pack);
    }
}

experimental::MemoryRequirements CpuSoftmaxGeneric::workspace() const
{
    return _aux_mem;
}

} // namespace cpu
} // namespace arm_compute
