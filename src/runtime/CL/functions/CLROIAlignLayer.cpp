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
#include "arm_compute/runtime/CL/functions/CLROIAlignLayer.h"

#include "arm_compute/core/CL/ICLArray.h"
#include "arm_compute/core/CL/kernels/CLROIAlignLayerKernel.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
Status CLROIAlignLayer::validate(const ITensorInfo *input, const ITensorInfo *rois, ITensorInfo *output, const ROIPoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(CLROIAlignLayerKernel::validate(input, rois, output, pool_info));

    return Status{};
}

void CLROIAlignLayer::configure(const ICLTensor *input, const ICLTensor *rois, ICLTensor *output, const ROIPoolingLayerInfo &pool_info)
{
    // Configure ROI pooling kernel
    auto k = arm_compute::support::cpp14::make_unique<CLROIAlignLayerKernel>();
    k->configure(input, rois, output, pool_info);
    _kernel = std::move(k);
}

} // namespace arm_compute
