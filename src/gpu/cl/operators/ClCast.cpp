/*
 * Copyright (c) 2021, 2024 Arm Limited.
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
#include "src/gpu/cl/operators/ClCast.h"

#include "src/common/utils/Log.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/kernels/ClCastKernel.h"

namespace arm_compute
{
namespace opencl
{
void ClCast::configure(const ClCompileContext &compile_context,
                       const ITensorInfo      *src,
                       ITensorInfo            *dst,
                       ConvertPolicy           policy)
{
    ARM_COMPUTE_LOG_PARAMS(src, dst, policy);
    ARM_COMPUTE_ERROR_THROW_ON(validate(src, dst, policy));

    auto k = std::make_unique<kernels::ClCastKernel>();
    k->configure(compile_context, src, dst, policy);
    _kernel = std::move(k);
}

Status ClCast::validate(const ITensorInfo *src, const ITensorInfo *dst, ConvertPolicy policy)
{
    // This operation mode is supported by ClCastKernel, however it has an unusual
    // casting behavior, which is not like casting between Int8 & UInt8. Therefore,
    // we do not expose this mode in the public api
    ARM_COMPUTE_RETURN_ERROR_ON(src->data_type() == DataType::QSYMM8_PER_CHANNEL &&
                                dst->data_type() == DataType::QASYMM8);

    return kernels::ClCastKernel::validate(src, dst, policy);
}
} // namespace opencl
} // namespace arm_compute
