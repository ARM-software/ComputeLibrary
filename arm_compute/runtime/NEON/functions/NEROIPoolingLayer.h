/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEROIPOOLINGLAYER_H__
#define __ARM_COMPUTE_NEROIPOOLINGLAYER_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/IArray.h"
#include "arm_compute/core/NEON/kernels/NEROIPoolingLayerKernel.h"

namespace arm_compute
{
class ITensor;

/** Basic function to run @ref NEROIPoolingLayerKernel.
 *
 * This function calls the following NEON kernels:
 * -# @ref NEROIPoolingLayerKernel
 *
 */
class NEROIPoolingLayer : public IFunction
{
public:
    /** Constructor */
    NEROIPoolingLayer();
    /** Set the input and output tensors.
     *
     * @param[in]  input     Source tensor. Data types supported: F32.
     * @param[in]  rois      Array containing @ref ROI.
     * @param[out] output    Destination tensor. Data types supported: Same as @p input.
     * @param[in]  pool_info Contains pooling operation information described in @ref ROIPoolingLayerInfo.
     *
     * @note The x and y dimensions of @p output tensor must be the same as that specified by @p pool_info 's pooled
     * width and pooled height.
     * @note The z dimensions of @p output tensor and @p input tensor must be the same.
     * @note The fourth dimension of @p output tensor must be the same as the number of elements in @p rois array.
     */
    void configure(const ITensor *input, const IROIArray *rois, ITensor *output, const ROIPoolingLayerInfo &pool_info);

    // Inherited methods overridden:
    void run() override;

private:
    NEROIPoolingLayerKernel _roi_kernel;
};
}
#endif /* __ARM_COMPUTE_NEROIPOOLINGLAYER_H__ */
