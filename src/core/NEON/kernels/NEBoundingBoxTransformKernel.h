/*
 * Copyright (c) 2019-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_NEBOUNDINGBOXTRANSFORMKERNEL_H
#define ARM_COMPUTE_NEBOUNDINGBOXTRANSFORMKERNEL_H

#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the bounding box kernel */
class NEBoundingBoxTransformKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEBoundingBoxTransformKernel";
    }

    /** Default constructor */
    NEBoundingBoxTransformKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEBoundingBoxTransformKernel(const NEBoundingBoxTransformKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEBoundingBoxTransformKernel &operator=(const NEBoundingBoxTransformKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEBoundingBoxTransformKernel(NEBoundingBoxTransformKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEBoundingBoxTransformKernel &operator=(NEBoundingBoxTransformKernel &&) = default;
    /** Default destructor */
    ~NEBoundingBoxTransformKernel() = default;

    /** Set the input and output tensors.
     *
     * @param[in]  boxes      Source tensor. Bounding box proposals in pixel coordinates. Size(M, 4), format [x1, y1, x2, y2]. Data types supported: QASYMM16/F16/F32.
     * @param[out] pred_boxes Destination tensor. Pixel coordinates of the transformed bounding boxes. Size (M, 4*K), format [x1, y1, x2, y2]. Data types supported: Same as @p input
     * @param[in]  deltas     Bounding box translations and scales. Size (M, 4*K), format [dx, dy, dw, dh], K  is the number of classes.
     *                        Data types supported: QASYMM8 if @p input is QASYMM16, otherwise same as @p input.
     * @param[in]  info       Contains BoundingBox operation information described in @ref BoundingBoxTransformInfo.
     *
     * @note Only single image prediction is supported. Height and Width (and scale) of the image will be contained in the BoundingBoxTransformInfo struct.
     *
     */
    void configure(const ITensor *boxes, ITensor *pred_boxes, const ITensor *deltas, const BoundingBoxTransformInfo &info);

    /** Static function to check if given info will lead to a valid configuration of @ref CLBoundingBoxTransform
     *
     * @param[in] boxes      Source tensor info. Bounding box proposals in pixel coordinates. Size(M, 4), format [x1, y1, x2, y2]. Data types supported: QASYMM16/F16/F32.
     * @param[in] pred_boxes Destination tensor info. Pixel coordinates of the transformed bounding boxes. Size (M, 4*K), format [x1, y1, x2, y2]. Data types supported: Same as @p input
     * @param[in] deltas     Bounding box translations and scales. Size (M, 4*K), format [dx, dy, dw, dh], K  is the number of classes.
     *                       Data types supported: QASYMM8 if @p input is QASYMM16, otherwise same as @p input.
     * @param[in] info       Contains BoundingBox operation information described in @ref BoundingBoxTransformInfo.
     *
     * @note Only single image prediction is supported. Height and Width (and scale) of the image will be contained in the BoundingBoxTransformInfo struct.
     *
     * @return a Status
     */
    static Status validate(const ITensorInfo *boxes, const ITensorInfo *pred_boxes, const ITensorInfo *deltas, const BoundingBoxTransformInfo &info);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    const ITensor           *_boxes;
    ITensor                 *_pred_boxes;
    const ITensor           *_deltas;
    BoundingBoxTransformInfo _bbinfo;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEBOUNDINGBOXTRANSFORMKERNEL_H */
