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
#include "arm_compute/runtime/CL/functions/CLBoundingBoxTransform.h"

#include "arm_compute/core/CL/kernels/CLBoundingBoxTransformKernel.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
void CLBoundingBoxTransform::configure(const ICLTensor *boxes, ICLTensor *pred_boxes, const ICLTensor *deltas, const BoundingBoxTransformInfo &info)
{
    // Configure Bounding Box kernel
    auto k = arm_compute::support::cpp14::make_unique<CLBoundingBoxTransformKernel>();
    k->configure(boxes, pred_boxes, deltas, info);
    _kernel = std::move(k);
}

Status CLBoundingBoxTransform::validate(const ITensorInfo *boxes, const ITensorInfo *pred_boxes, const ITensorInfo *deltas, const BoundingBoxTransformInfo &info)
{
    return CLBoundingBoxTransformKernel::validate(boxes, pred_boxes, deltas, info);
}
} // namespace arm_compute
