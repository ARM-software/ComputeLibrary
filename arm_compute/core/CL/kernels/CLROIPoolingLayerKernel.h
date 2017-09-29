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
#ifndef __ARM_COMPUTE_CLROIPOOLINGLAYERKERNEL_H__
#define __ARM_COMPUTE_CLROIPOOLINGLAYERKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

#include "arm_compute/core/CL/ICLArray.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the ROI pooling layer kernel */
class CLROIPoolingLayerKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLROIPoolingLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLROIPoolingLayerKernel(const CLROIPoolingLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLROIPoolingLayerKernel &operator=(const CLROIPoolingLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLROIPoolingLayerKernel(CLROIPoolingLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLROIPoolingLayerKernel &operator=(CLROIPoolingLayerKernel &&) = default;
    /** Default destructor */
    ~CLROIPoolingLayerKernel() = default;

    /** Set the input and output tensors.
     *
     * @param[in]  input     Source tensor. Data types supported: F16/F32.
     * @param[in]  rois      Array containing @ref ROI.
     * @param[out] output    Destination tensor. Data types supported: Same as @p input.
     * @param[in]  pool_info Contains pooling operation information described in @ref ROIPoolingLayerInfo.
     *
     * @note The x and y dimensions of @p output tensor must be the same as @p pool_info 's pooled
     * width and pooled height.
     * @note The z dimensions of @p output tensor and @p input tensor must be the same.
     * @note The fourth dimension of @p output tensor must be the same as the number of elements in @p rois array.
     */
    void configure(const ICLTensor *input, const ICLROIArray *rois, ICLTensor *output, const ROIPoolingLayerInfo &pool_info);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor    *_input;
    const ICLROIArray *_rois;
    ICLTensor          *_output;
    ROIPoolingLayerInfo _pool_info;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLROIPOOLINGLAYERKERNEL_H__ */
