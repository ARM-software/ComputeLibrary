/*
 * Copyright (c) 2019-2021, 2024-2025 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEBoundingBoxTransform.h"

#include "arm_compute/core/Validate.h"

#include "src/common/utils/Log.h"
#include "src/common/utils/profile/acl_profile.h"
#include "src/core/NEON/kernels/NEBoundingBoxTransformKernel.h"

namespace arm_compute
{
void NEBoundingBoxTransform::configure(const ITensor                  *boxes,
                                       ITensor                        *pred_boxes,
                                       const ITensor                  *deltas,
                                       const BoundingBoxTransformInfo &info)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "NEBoundingBoxTransform::configure");
    ARM_COMPUTE_LOG_PARAMS(boxes, pred_boxes, deltas, info);
    // Configure Bounding Box kernel
    auto k = std::make_unique<NEBoundingBoxTransformKernel>();
    k->configure(boxes, pred_boxes, deltas, info);
    _kernel = std::move(k);
}

Status NEBoundingBoxTransform::validate(const ITensorInfo              *boxes,
                                        const ITensorInfo              *pred_boxes,
                                        const ITensorInfo              *deltas,
                                        const BoundingBoxTransformInfo &info)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "NEBoundingBoxTransform::validate");
    ARM_COMPUTE_RETURN_ERROR_ON_DYNAMIC_SHAPE(boxes, pred_boxes, deltas);
    return NEBoundingBoxTransformKernel::validate(boxes, pred_boxes, deltas, info);
}
} // namespace arm_compute
