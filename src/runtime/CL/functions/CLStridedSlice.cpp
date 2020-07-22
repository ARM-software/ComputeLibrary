/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLStridedSlice.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/kernels/CLStridedSliceKernel.h"
#include "arm_compute/core/Types.h"
#include "support/MemorySupport.h"

namespace arm_compute
{
namespace experimental
{
void CLStridedSlice::configure(const CLCompileContext &compile_context, const ITensorInfo *input, ITensorInfo *output,
                               const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                               int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask)
{
    auto k = arm_compute::support::cpp14::make_unique<CLStridedSliceKernel>();
    k->configure(compile_context, input, output, starts, ends, strides, begin_mask, end_mask, shrink_axis_mask);
    _kernel = std::move(k);
}

Status CLStridedSlice::validate(const ITensorInfo *input, const ITensorInfo *output,
                                const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                                int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask)
{
    return CLStridedSliceKernel::validate(input, output, starts, ends, strides, begin_mask, end_mask, shrink_axis_mask);
}
} // namespace experimental

struct CLStridedSlice::Impl
{
    const ICLTensor                              *src{ nullptr };
    ICLTensor                                    *dst{ nullptr };
    CLRuntimeContext                             *ctx{ nullptr };
    std::unique_ptr<experimental::CLStridedSlice> op{ nullptr };
};

CLStridedSlice::CLStridedSlice(CLRuntimeContext *ctx)
    : _impl(support::cpp14::make_unique<Impl>())
{
    _impl->ctx = ctx;
}

CLStridedSlice::CLStridedSlice(CLStridedSlice &&) = default;
CLStridedSlice &CLStridedSlice::operator=(CLStridedSlice &&) = default;
CLStridedSlice::~CLStridedSlice()                            = default;

void CLStridedSlice::configure(const ICLTensor *input, ICLTensor *output,
                               const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                               int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, starts, ends, strides, begin_mask, end_mask, shrink_axis_mask);
}

void CLStridedSlice::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output,
                               const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                               int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input);

    _impl->src = input;
    _impl->dst = output;

    _impl->op = arm_compute::support::cpp14::make_unique<experimental::CLStridedSlice>();
    _impl->op->configure(compile_context, _impl->src->info(), _impl->dst->info(), starts, ends, strides, begin_mask, end_mask, shrink_axis_mask);
}

Status CLStridedSlice::validate(const ITensorInfo *input, const ITensorInfo *output,
                                const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                                int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask)
{
    return experimental::CLStridedSlice::validate(input, output, starts, ends, strides, begin_mask, end_mask, shrink_axis_mask);
}

void CLStridedSlice::run()
{
    const InputTensorMap  src{ { TensorType::ACL_SRC, _impl->src } };
    const OutputTensorMap dst{ { TensorType::ACL_DST, _impl->dst } };

    _impl->op->run(src, dst, {});
}
} // namespace arm_compute
