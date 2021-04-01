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
#ifndef ARM_COMPUTE_CLROIPOOLINGLAYERKERNEL_H
#define ARM_COMPUTE_CLROIPOOLINGLAYERKERNEL_H

#include "src/core/CL/ICLKernel.h"
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
     * @param[in]  rois      ROIs tensor, it is a 2D tensor of size [5, N] (where N is the number of ROIs) containing top left and bottom right corner
     *                       as coordinate of an image and batch_id of ROI [ batch_id, x1, y1, x2, y2 ]. Data types supported: U16
     * @param[out] output    Destination tensor. Data types supported: Same as @p input.
     * @param[in]  pool_info Contains pooling operation information described in @ref ROIPoolingLayerInfo.
     *
     * @note The x and y dimensions of @p output tensor must be the same as @p pool_info 's pooled
     * width and pooled height.
     * @note The z dimensions of @p output tensor and @p input tensor must be the same.
     * @note The fourth dimension of @p output tensor must be the same as the number of elements in @p rois array.
     */
    void configure(const ICLTensor *input, const ICLTensor *rois, ICLTensor *output, const ROIPoolingLayerInfo &pool_info);
    /** Set the input and output tensors.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. Data types supported: F16/F32/QASYMM8
     * @param[in]  rois            ROIs tensor, it is a 2D tensor of size [5, N] (where N is the number of ROIs) containing top left and bottom right corner
     *                             as coordinate of an image and batch_id of ROI [ batch_id, x1, y1, x2, y2 ]. Data types supported: U16
     * @param[out] output          Destination tensor. Data types supported: Same as @p input.
     * @param[in]  pool_info       Contains pooling operation information described in @ref ROIPoolingLayerInfo.
     *
     * @note The x and y dimensions of @p output tensor must be the same as @p pool_info 's pooled
     * width and pooled height.
     * @note The z dimensions of @p output tensor and @p input tensor must be the same.
     * @note The fourth dimension of @p output tensor must be the same as the number of elements in @p rois array.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *rois, const ICLTensor *output, const ROIPoolingLayerInfo &pool_info);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

    /** Static Validate function to check inputs will lead to valid configuration of @ref CLROIPoolingLayer
     *
     * @param[in]  input     Source tensor. Data types supported: F16/F32/QASYMM8
     * @param[in]  rois      ROIs tensor, it is a 2D tensor of size [5, N] (where N is the number of ROIs) containing top left and bottom right corner
     *                       as coordinate of an image and batch_id of ROI [ batch_id, x1, y1, x2, y2 ]. Data types supported: U16
     * @param[out] output    Destination tensor. Data types supported: Same as @p input.
     * @param[in]  pool_info Contains pooling operation information described in @ref ROIPoolingLayerInfo.
     *
     * @note The x and y dimensions of @p output tensor must be the same as @p pool_info 's pooled
     * width and pooled height.
     * @note The z dimensions of @p output tensor and @p input tensor must be the same.
     * @note The fourth dimension of @p output tensor must be the same as the number of elements in @p rois array.
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *rois, const ITensorInfo *output, const ROIPoolingLayerInfo &pool_info);

private:
    const ICLTensor    *_input;
    const ICLTensor    *_rois;
    const ICLTensor    *_output;
    ROIPoolingLayerInfo _pool_info;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLROIPOOLINGLAYERKERNEL_H */
