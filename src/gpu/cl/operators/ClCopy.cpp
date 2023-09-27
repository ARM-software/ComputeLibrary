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
#include "src/gpu/cl/operators/ClCopy.h"

#include "src/common/utils/Log.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/kernels/ClCopyKernel.h"

namespace arm_compute
{
namespace opencl
{
void ClCopy::configure(const ClCompileContext &compile_context, ITensorInfo *src, ITensorInfo *dst, Window *dst_window)
{
    ARM_COMPUTE_LOG_PARAMS(src, dst, dst_window);
    auto k = std::make_unique<kernels::ClCopyKernel>();
    k->configure(compile_context, src, dst, dst_window);
    _kernel = std::move(k);
}

Status ClCopy::validate(const ITensorInfo *src, const ITensorInfo *dst, Window *dst_window)
{
    return kernels::ClCopyKernel::validate(src, dst, dst_window);
}
} // namespace opencl
} // namespace arm_compute
