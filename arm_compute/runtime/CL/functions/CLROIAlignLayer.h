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
#ifndef ARM_COMPUTE_CLROIALIGNLAYER_H
#define ARM_COMPUTE_CLROIALIGNLAYER_H

#include "arm_compute/core/CL/ICLArray.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;
class ROIPoolingLayerInfo;
class ITensorInfo;

/** Basic function to run @ref CLROIAlignLayerKernel.
 *
 * This function calls the following OpenCL kernels:
 * -# @ref CLROIAlignLayerKernel
 *
 */
class CLROIAlignLayer : public ICLSimpleFunction
{
public:
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0           |src1           |dst            |
     * |:--------------|:--------------|:--------------|
     * |F16            |F16            |F16            |
     * |F32            |F32            |F32            |
     * |QASYMM8        |QASYMM16       |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM16       |QASYMM8_SIGNED |
     *
     * @param[in]  input     Source tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  rois      ROIs tensor, it is a 2D tensor of size [5, N] (where N is the number of ROIs) containing top left and bottom right corner
     *                       as coordinate of an image and batch_id of ROI [ batch_id, x1, y1, x2, y2 ].
     *                       Data types supported: QASYMM16 with scale of 0.125 and 0 offset if @p input is QASYMM8/QASYMM8_SIGNED, otherwise same as @p input
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
     * @param[in]  input           Source tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  rois            ROIs tensor, it is a 2D tensor of size [5, N] (where N is the number of ROIs) containing top left and bottom right corner
     *                             as coordinate of an image and batch_id of ROI [ batch_id, x1, y1, x2, y2 ].
     *                             Data types supported: QASYMM16 with scale of 0.125 and 0 offset if @p input is QASYMM8/QASYMM8_SIGNED, otherwise same as @p input
     * @param[out] output          Destination tensor. Data types supported: Same as @p input.
     * @param[in]  pool_info       Contains pooling operation information described in @ref ROIPoolingLayerInfo.
     *
     * @note The x and y dimensions of @p output tensor must be the same as @p pool_info 's pooled
     * width and pooled height.
     * @note The z dimensions of @p output tensor and @p input tensor must be the same.
     * @note The fourth dimension of @p output tensor must be the same as the number of elements in @p rois array.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *rois, ICLTensor *output, const ROIPoolingLayerInfo &pool_info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLROIAlignLayer
     *
     * @param[in] input     Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in] rois      ROIs tensor info. Data types supported: QASYMM16 with scale of 0.125 and 0 offset if @p input is QASYMM8/QASYMM8_SIGNED,
     *                      otherwise same as @p input
     * @param[in] output    Destination tensor info. Data types supported: Same as @p input.
     * @param[in] pool_info Contains pooling operation information described in @ref ROIPoolingLayerInfo.
     *
     * @note The x and y dimensions of @p output tensor must be the same as @p pool_info 's pooled
     * width and pooled height.
     * @note The z dimensions of @p output tensor and @p input tensor must be the same.
     * @note The fourth dimension of @p output tensor must be the same as the number of elements in @p rois array.
     *
     * @return a Status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *rois, ITensorInfo *output, const ROIPoolingLayerInfo &pool_info);
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLROIALIGNLAYER_H */
