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
#include "src/runtime/gpu/cl/operators/ClPixelWiseMultiplication.h"

#include "arm_compute/runtime/CL/CLScheduler.h"
#include "src/core/CL/kernels/CLFillBorderKernel.h"
#include "src/core/gpu/cl/ClCompileContext.h"
#include "src/core/gpu/cl/kernels/ClPixelWiseMultiplicationKernel.h"

namespace arm_compute
{
namespace opencl
{
namespace
{
ITensorPack select_border_input(ITensorPack &tensors)
{
    ITensorPack pack;
    if(tensors.get_tensor(TensorType::ACL_DST)->info()->dimension(0) > 1)
    {
        if(tensors.get_const_tensor(TensorType::ACL_SRC_1)->info()->dimension(0) == 1)
        {
            pack.add_tensor(TensorType::ACL_SRC, tensors.get_const_tensor(TensorType::ACL_SRC_1));
        }
        else
        {
            pack.add_tensor(TensorType::ACL_SRC, tensors.get_const_tensor(TensorType::ACL_SRC_0));
        }
    }
    return pack;
}
} // namespace

void ClPixelWiseMultiplication::configure(const CLCompileContext &compile_context, ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst, float scale,
                                          ConvertPolicy overflow_policy, RoundingPolicy rounding_policy, const ActivationLayerInfo &act_info)
{
    auto k = std::make_unique<kernels::ClPixelWiseMultiplicationKernel>();
    k->configure(compile_context, src1, src2, dst, scale, overflow_policy, rounding_policy, act_info);
    _kernel = std::move(k);

    if(dst->dimension(0) > 1)
    {
        ITensorInfo *broadcasted_info = (src1->dimension(0) == 1) ? src1 : src2;

        if(broadcasted_info->dimension(0) == 1)
        {
            auto b = std::make_unique<CLFillBorderKernel>();
            b->configure(compile_context, broadcasted_info, _kernel->border_size(), BorderMode::REPLICATE);
            _border_handler = std::move(b);
        }
    }
}

Status ClPixelWiseMultiplication::validate(const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, float scale,
                                           ConvertPolicy overflow_policy, RoundingPolicy rounding_policy, const ActivationLayerInfo &act_info)
{
    return kernels::ClPixelWiseMultiplicationKernel::validate(src1, src2, dst, scale, overflow_policy, rounding_policy, act_info);
}

void ClPixelWiseMultiplication::run(ITensorPack &tensors)
{
    if(_border_handler)
    {
        auto border_pack = select_border_input(tensors);
        CLScheduler::get().enqueue_op(*_border_handler, border_pack);
    }
    ICLOperator::run(tensors);
}

void ClComplexPixelWiseMultiplication::configure(const CLCompileContext &compile_context, ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst, const ActivationLayerInfo &act_info)
{
    auto k = std::make_unique<kernels::ClComplexPixelWiseMultiplicationKernel>();
    k->configure(compile_context, src1, src2, dst, act_info);
    _kernel = std::move(k);

    if(dst->dimension(0) > 1)
    {
        ITensorInfo *broadcasted_info = (src1->dimension(0) == 1) ? src1 : src2;

        if(broadcasted_info->dimension(0) == 1)
        {
            auto b = std::make_unique<CLFillBorderKernel>();
            b->configure(compile_context, broadcasted_info, _kernel->border_size(), BorderMode::REPLICATE);
            _border_handler = std::move(b);
        }
    }
}

Status ClComplexPixelWiseMultiplication::validate(const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, const ActivationLayerInfo &act_info)
{
    return kernels::ClComplexPixelWiseMultiplicationKernel::validate(src1, src2, dst, act_info);
}

void ClComplexPixelWiseMultiplication::run(ITensorPack &tensors)
{
    if(_border_handler)
    {
        auto border_pack = select_border_input(tensors);
        CLScheduler::get().enqueue_op(*_border_handler, border_pack);
    }
    ICLOperator::run(tensors);
}
} // namespace opencl
} // namespace arm_compute