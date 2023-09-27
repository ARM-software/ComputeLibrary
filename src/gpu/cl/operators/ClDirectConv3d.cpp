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
#include "src/gpu/cl/operators/ClDirectConv3d.h"

#include "arm_compute/runtime/CL/CLScheduler.h"

#include "src/gpu/cl/kernels/ClDirectConv3dKernel.h"

namespace arm_compute
{
namespace opencl
{
void ClDirectConv3d::configure(const CLCompileContext &compile_context,
                               const ITensorInfo      *src0,
                               const ITensorInfo      *src1,
                               const ITensorInfo      *src2,
                               ITensorInfo            *dst,
                               const Conv3dInfo       &conv3d_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src0);

    // Configure direct convolution 3d kernel
    auto k = std::make_unique<kernels::ClDirectConv3dKernel>();
    k->configure(compile_context, src0, src1, src2, dst, conv3d_info);
    _direct_conv3d_kernel = std::move(k);
}

Status ClDirectConv3d::validate(const ITensorInfo *src0,
                                const ITensorInfo *src1,
                                const ITensorInfo *src2,
                                const ITensorInfo *dst,
                                const Conv3dInfo  &conv3d_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(kernels::ClDirectConv3dKernel::validate(src0, src1, src2, dst, conv3d_info));
    return Status{};
}

void ClDirectConv3d::run(ITensorPack &tensors)
{
    // Run direct convolution 3d
    CLScheduler::get().enqueue_op(*_direct_conv3d_kernel.get(), tensors, true);
}
} // namespace opencl
} // namespace arm_compute
