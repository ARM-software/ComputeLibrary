/*
 * Copyright (c) 2024 Arm Limited.
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
#include "src/gpu/cl/kernels/ClScatterKernel.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/TensorInfo.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
ClScatterKernel::ClScatterKernel()
{
}

Status ClScatterKernel::validate(const ITensorInfo *src,
                                 const ITensorInfo *updates,
                                 const ITensorInfo *indices,
                                 const ITensorInfo *dst,
                                 const ScatterInfo &info)
{
    ARM_COMPUTE_UNUSED(src);
    ARM_COMPUTE_UNUSED(updates);
    ARM_COMPUTE_UNUSED(indices);
    ARM_COMPUTE_UNUSED(dst);
    ARM_COMPUTE_UNUSED(info);

    return Status{};
}
void ClScatterKernel::configure(const ClCompileContext &compile_context,
                                const ITensorInfo      *src,
                                const ITensorInfo      *updates,
                                const ITensorInfo      *indices,
                                ITensorInfo            *dst,
                                const ScatterInfo      &info)
{
    ARM_COMPUTE_UNUSED(compile_context);
    ARM_COMPUTE_UNUSED(src);
    ARM_COMPUTE_UNUSED(updates);
    ARM_COMPUTE_UNUSED(indices);
    ARM_COMPUTE_UNUSED(dst);
    ARM_COMPUTE_UNUSED(info);
}

void ClScatterKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_UNUSED(tensors);
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_UNUSED(queue);
}

} // namespace kernels
} // namespace opencl
} // namespace arm_compute
