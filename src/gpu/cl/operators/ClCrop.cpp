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
#include "src/gpu/cl/operators/ClCrop.h"

#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/kernels/ClCropKernel.h"

#include "src/common/utils/Log.h"

namespace arm_compute
{
namespace opencl
{
void ClCrop::configure(const ClCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst, Coordinates2D start, Coordinates2D end, uint32_t batch_index, float extrapolation_value,
                       Window *dst_window)
{
    ARM_COMPUTE_LOG_PARAMS(src, dst, start, end, batch_index, extrapolation_value, dst_window);
    auto k = std::make_unique<kernels::ClCropKernel>();
    k->configure(compile_context, src, dst, start, end, batch_index, extrapolation_value, dst_window);
    _kernel = std::move(k);
}

Status ClCrop::validate(const ITensorInfo *src, const ITensorInfo *dst, Coordinates2D start, Coordinates2D end, uint32_t batch_index, float extrapolation_value, Window *dst_window)
{
    return kernels::ClCropKernel::validate(src, dst, start, end, batch_index, extrapolation_value, dst_window);
}
} // namespace opencl
} // namespace arm_compute