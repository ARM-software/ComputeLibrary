/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLStridedSliceKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/helpers/tensor_transform.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
void CLSlice::configure(const ICLTensor *input, ICLTensor *output, const Coordinates &starts, const Coordinates &ends)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input);

    // Get absolute end coordinates
    const int32_t slice_end_mask = arm_compute::helpers::tensor_transform::construct_slice_end_mask(ends);

    auto k = arm_compute::support::cpp14::make_unique<CLStridedSliceKernel>();
    k->configure(input, output, starts, ends, BiStrides(), 0, slice_end_mask, 0);
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
} // namespace arm_compute
