/*
 * Copyright (c) 2018-2021, 2024-2025 Arm Limited.
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
#ifndef ACL_ARM_COMPUTE_RUNTIME_CL_FUNCTIONS_CLBOUNDINGBOXTRANSFORM_H
#define ACL_ARM_COMPUTE_RUNTIME_CL_FUNCTIONS_CLBOUNDINGBOXTRANSFORM_H

/** @file
 * @publicapi
 */

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

namespace arm_compute
{
class CLCompileContext;
class CLBoundingBoxTransformKernel;
class BoundingBoxTransformInfo;
class ICLTensor;
class ITensorInfo;

/** Basic function to run CLBoundingBoxTransformKernel.
 *
 * This function calls the following OpenCL kernels:
 * -# CLBoundingBoxTransformKernel
 */
class CLBoundingBoxTransform : public ICLSimpleFunction
{
public:
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src0           |src1           |dst            |
     * |:--------------|:--------------|:--------------|
     * |QASYMM16       |QASYMM8        |QASYMM16       |
     * |F16            |F16            |F16            |
     * |F32            |F32            |F32            |
     *
     * @param[in]  boxes      Source tensor. Bounding box proposals in pixel coordinates. Size(M, 4), format [x1, y1, x2, y2]. Data types supported: QASYMM16/F16/F32.
     * @param[out] pred_boxes Destination tensor. Pixel coordinates of the transformed bounding boxes. Size (M, 4*K), format [x1, y1, x2, y2]. Data types supported: Same as @p input
     * @param[in]  deltas     Bounding box translations and scales. Size (M, 4*K), format [dx, dy, dw, dh], K  is the number of classes.
     *                        Data types supported: QASYMM8 if @p input is QASYMM16, otherwise same as @p input
     * @param[in]  info       Contains BoundingBox operation information described in @ref BoundingBoxTransformInfo.
     *
     * @note Only single image prediction is supported. Height and Width (and scale) of the image will be contained in the BoundingBoxTransformInfo struct.
     */
    void configure(const ICLTensor                *boxes,
                   ICLTensor                      *pred_boxes,
                   const ICLTensor                *deltas,
                   const BoundingBoxTransformInfo &info);
    /** Set the input and output tensors.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  boxes           Source tensor. Bounding box proposals in pixel coordinates. Size(M, 4), format [x1, y1, x2, y2]. Data types supported: QASYMM16/F16/F32.
     * @param[out] pred_boxes      Destination tensor. Pixel coordinates of the transformed bounding boxes. Size (M, 4*K), format [x1, y1, x2, y2]. Data types supported: Same as @p input
     * @param[in]  deltas          Bounding box translations and scales. Size (M, 4*K), format [dx, dy, dw, dh], K  is the number of classes.
     *                             Data types supported: QASYMM8 if @p input is QASYMM16, otherwise same as @p input
     * @param[in]  info            Contains BoundingBox operation information described in @ref BoundingBoxTransformInfo.
     *
     * @note Only single image prediction is supported. Height and Width (and scale) of the image will be contained in the BoundingBoxTransformInfo struct.
     */
    void configure(const CLCompileContext         &compile_context,
                   const ICLTensor                *boxes,
                   ICLTensor                      *pred_boxes,
                   const ICLTensor                *deltas,
                   const BoundingBoxTransformInfo &info);

    /** Static function to check if given info will lead to a valid configuration of @ref CLBoundingBoxTransform
     *
     * @param[in] boxes      Source tensor info. Bounding box proposals in pixel coordinates. Size(M, 4), format [x1, y1, x2, y2]. Data types supported: QASYMM16/F16/F32.
     * @param[in] pred_boxes Destination tensor info. Pixel coordinates of the transformed bounding boxes. Size (M, 4*K), format [x1, y1, x2, y2]. Data types supported: Same as @p input
     * @param[in] deltas     Bounding box translations and scales. Size (M, 4*K), format [dx, dy, dw, dh], K  is the number of classes.
     *                       Data types supported: QASYMM8 if @p input is QASYMM16, otherwise same as @p input
     * @param[in] info       Contains BoundingBox operation information described in @ref BoundingBoxTransformInfo.
     *
     * @note Only single image prediction is supported. Height and Width (and scale) of the image will be contained in the BoundingBoxTransformInfo struct.
     *
     * @return a Status
     */
    static Status validate(const ITensorInfo              *boxes,
                           const ITensorInfo              *pred_boxes,
                           const ITensorInfo              *deltas,
                           const BoundingBoxTransformInfo &info);
};
} // namespace arm_compute
#endif // ACL_ARM_COMPUTE_RUNTIME_CL_FUNCTIONS_CLBOUNDINGBOXTRANSFORM_H
