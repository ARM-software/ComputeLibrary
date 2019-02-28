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

#include "arm_compute/runtime/CL/functions/CLArgMinMaxLayer.h"

#include "arm_compute/core/CL/kernels/CLReductionOperationKernel.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

namespace arm_compute
{
void CLArgMinMaxLayer::configure(const ICLTensor *input, int axis, ICLTensor *output, const ReductionOperation &op)
{
    auto k = arm_compute::support::cpp14::make_unique<CLReductionOperationKernel>();
    k->configure(input, output, axis, op);
    _kernel = std::move(k);
}

Status CLArgMinMaxLayer::validate(const ITensorInfo *input, int axis, const ITensorInfo *output, const ReductionOperation &op)
{
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(op != ReductionOperation::ARG_IDX_MAX && op != ReductionOperation::ARG_IDX_MIN, "Invalid operation");
    return CLReductionOperationKernel::validate(input, output, axis, op);
}
} // namespace arm_compute