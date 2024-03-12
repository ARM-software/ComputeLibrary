/*
 * Copyright (c) 2021, 2023 Arm Limited.
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

#include "arm_compute/core/experimental/Types.h"
#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include "src/common/utils/Log.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/gpu/cl/kernels/ClSoftmaxKernel.h"
#include "src/gpu/cl/utils/ClAuxTensorHandler.h"

using namespace arm_compute::experimental;

namespace arm_compute
{
namespace opencl
{

ClSoftmax::ClSoftmax() : _aux_mem(InternalTensorIdx::COUNT)
{
}

void ClSoftmax::configure(const CLCompileContext  &compile_context,
                          const ITensorInfo       &src,
                          ITensorInfo             &dst,
                          const SoftmaxKernelInfo &info)
{
    ARM_COMPUTE_LOG_PARAMS(src, dst, info);

    auto k = std::make_unique<kernels::ClSoftmaxKernel>();
    k->configure(compile_context, src, dst, info);

    _tmp_info = k->tmp_tensor_info();

    _kernel = std::move(k);

    _aux_mem[InternalTensorIdx::TMP] =
        MemoryInfo(offset_int_vec(InternalTensorIdx::TMP), MemoryLifetime::Temporary, _tmp_info.total_size());
}

Status ClSoftmax::validate(const ITensorInfo &src, const ITensorInfo &dst, const SoftmaxKernelInfo &info)
{
    return kernels::ClSoftmaxKernel::validate(src, dst, info);
}

void ClSoftmax::run(ITensorPack &tensors)
{
    CLAuxTensorHandler tmp(offset_int_vec(InternalTensorIdx::TMP), _tmp_info, tensors);

    tensors.add_tensor(TensorType::ACL_INT_0, tmp.get());

    CLScheduler::get().enqueue_op(*_kernel, tensors, false);
}

experimental::MemoryRequirements ClSoftmax::workspace() const
{
    return _aux_mem;
}

} // namespace opencl
} // namespace arm_compute
