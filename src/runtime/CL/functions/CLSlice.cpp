/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLSlice.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/helpers/tensor_transform.h"
#include "src/core/CL/kernels/CLStridedSliceKernel.h"

#include "src/common/utils/Log.h"

namespace arm_compute
{
namespace experimental
{
void CLSlice::configure(const CLCompileContext &compile_context, const ITensorInfo *input, ITensorInfo *output, const Coordinates &starts, const Coordinates &ends)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input);
    ARM_COMPUTE_LOG_PARAMS(input, output, starts, ends);

    // Get absolute end coordinates
    const int32_t slice_end_mask = arm_compute::helpers::tensor_transform::construct_slice_end_mask(ends);

    auto k = std::make_unique<CLStridedSliceKernel>();
    k->configure(compile_context, input, output, starts, ends, BiStrides(), 0, slice_end_mask, 0);
    _kernel = std::move(k);
}

Status CLSlice::validate(const ITensorInfo *input, const ITensorInfo *output, const Coordinates &starts, const Coordinates &ends)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);

    // Check start dimensions for being non-negative
    ARM_COMPUTE_RETURN_ERROR_ON(std::any_of(starts.cbegin(), starts.cbegin() + starts.num_dimensions(), [](int i)
    {
        return i < 0;
    }));

    // Get absolute end coordinates
    const int32_t slice_end_mask = arm_compute::helpers::tensor_transform::construct_slice_end_mask(ends);

    return CLStridedSliceKernel::validate(input, output, starts, ends, BiStrides(), 0, slice_end_mask, 0);
}
} // namespace experimental

struct CLSlice::Impl
{
    const ICLTensor                       *src{ nullptr };
    ICLTensor                             *dst{ nullptr };
    std::unique_ptr<experimental::CLSlice> op{ nullptr };
};

CLSlice::CLSlice()
    : _impl(std::make_unique<Impl>())
{
}
CLSlice::CLSlice(CLSlice &&) = default;
CLSlice &CLSlice::operator=(CLSlice &&) = default;
CLSlice::~CLSlice()                     = default;

Status CLSlice::validate(const ITensorInfo *input, const ITensorInfo *output, const Coordinates &starts, const Coordinates &ends)
{
    return experimental::CLSlice::validate(input, output, starts, ends);
}

void CLSlice::configure(const ICLTensor *input, ICLTensor *output, const Coordinates &starts, const Coordinates &ends)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, starts, ends);
}

void CLSlice::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, const Coordinates &starts, const Coordinates &ends)
{
    _impl->src = input;
    _impl->dst = output;
    _impl->op  = std::make_unique<experimental::CLSlice>();
    _impl->op->configure(compile_context, input->info(), output->info(), starts, ends);
}

void CLSlice::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}
} // namespace arm_compute
