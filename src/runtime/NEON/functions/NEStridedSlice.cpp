/*
 * Copyright (c) 2018-2020 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEStridedSlice.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/kernels/NEStridedSliceKernel.h"
#include "arm_compute/core/Types.h"
#include "support/MemorySupport.h"

namespace arm_compute
{
namespace experimental
{
void NEStridedSlice::configure(const ITensorInfo *input, ITensorInfo *output,
                               const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                               int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask)
{
    auto k = arm_compute::support::cpp14::make_unique<NEStridedSliceKernel>();
    k->configure(input, output, starts, ends, strides, begin_mask, end_mask, shrink_axis_mask);
    _kernel = std::move(k);
}

Status NEStridedSlice::validate(const ITensorInfo *input, const ITensorInfo *output,
                                const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                                int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask)
{
    return NEStridedSliceKernel::validate(input, output, starts, ends, strides, begin_mask, end_mask, shrink_axis_mask);
}

MemoryRequirements NEStridedSlice::workspace() const
{
    return MemoryRequirements{};
}
} // namespace experimental

struct NEStridedSlice::Impl
{
    const ITensor                                *src{ nullptr };
    ITensor                                      *dst{ nullptr };
    std::unique_ptr<experimental::NEStridedSlice> op{ nullptr };
};

NEStridedSlice::NEStridedSlice()
    : _impl(support::cpp14::make_unique<Impl>())
{
}
NEStridedSlice::NEStridedSlice(NEStridedSlice &&) = default;
NEStridedSlice &NEStridedSlice::operator=(NEStridedSlice &&) = default;
NEStridedSlice::~NEStridedSlice()                            = default;

void NEStridedSlice::configure(const ITensor *input, ITensor *output,
                               const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                               int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask)
{
    _impl->src = input;
    _impl->dst = output;
    _impl->op  = arm_compute::support::cpp14::make_unique<experimental::NEStridedSlice>();
    _impl->op->configure(input->info(), output->info(), starts, ends, strides, begin_mask, end_mask, shrink_axis_mask);
}

void NEStridedSlice::run()
{
    const InputTensorMap  src{ { TensorType::ACL_SRC, _impl->src } };
    const OutputTensorMap dst{ { TensorType::ACL_DST, _impl->dst } };
    _impl->op->run(src, dst, {});
}

Status NEStridedSlice::validate(const ITensorInfo *input, const ITensorInfo *output,
                                const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                                int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask)
{
    return experimental::NEStridedSlice::validate(input, output, starts, ends, strides, begin_mask, end_mask, shrink_axis_mask);
}
} // namespace arm_compute
