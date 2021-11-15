/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEROIPoolingLayer.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/common/utils/Log.h"
#include "src/core/NEON/kernels/NEROIPoolingLayerKernel.h"

namespace arm_compute
{
NEROIPoolingLayer::~NEROIPoolingLayer() = default;

NEROIPoolingLayer::NEROIPoolingLayer()
    : _roi_kernel()
{
}

Status NEROIPoolingLayer::validate(const ITensorInfo *input, const ITensorInfo *rois, const ITensorInfo *output, const ROIPoolingLayerInfo &pool_info)
{
    return NEROIPoolingLayerKernel::validate(input, rois, output, pool_info);
}

void NEROIPoolingLayer::configure(const ITensor *input, const ITensor *rois, const ITensor *output, const ROIPoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_LOG_PARAMS(input, rois, output, pool_info);

    _roi_kernel = std::make_unique<NEROIPoolingLayerKernel>();
    _roi_kernel->configure(input, rois, output, pool_info);
}

void NEROIPoolingLayer::run()
{
    NEScheduler::get().schedule(_roi_kernel.get(), Window::DimX);
}
} // namespace arm_compute