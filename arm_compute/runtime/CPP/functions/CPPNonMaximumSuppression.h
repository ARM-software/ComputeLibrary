/*
 * Copyright (c) 2019 Arm Limited.
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
#ifndef ARM_COMPUTE_CPP_NONMAXIMUMSUPPRESSION_LAYER_H
#define ARM_COMPUTE_CPP_NONMAXIMUMSUPPRESSION_LAYER_H

#include "arm_compute/runtime/CPP/ICPPSimpleFunction.h"

#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ITensor;

/** CPP Function to perform non maximum suppression on the bounding boxes and scores
 *
 */
class CPPNonMaximumSuppression : public ICPPSimpleFunction
{
public:
    /** Configure the function to perform non maximal suppression
     *
     * @param[in]  bboxes          The input bounding boxes. Data types supported: F32.
     * @param[in]  scores          The corresponding input confidence. Same as @p bboxes.
     * @param[out] indices         The kept indices of bboxes after nms. Data types supported: S32.
     * @param[in]  max_output_size An integer tensor representing the maximum number of boxes to be selected by non max suppression.
     * @param[in]  score_threshold The threshold used to filter detection results.
     * @param[in]  nms_threshold   The threshold used in non maximum suppression.
     *
     */
    void configure(const ITensor *bboxes, const ITensor *scores, ITensor *indices, unsigned int max_output_size, const float score_threshold, const float nms_threshold);

    /** Static function to check if given arguments will lead to a valid configuration of @ref CPPNonMaximumSuppression
     *
     * @param[in]  bboxes          The input bounding boxes tensor info. Data types supported: F32.
     * @param[in]  scores          The corresponding input confidence tensor info. Same as @p bboxes.
     * @param[out] indices         The kept indices of bboxes after nms tensor info. Data types supported: S32.
     * @param[in]  max_output_size An integer tensor representing the maximum number of boxes to be selected by non max suppression.
     * @param[in]  score_threshold The threshold used to filter detection results.
     * @param[in]  nms_threshold   The threshold used in non maximum suppression.
     *
     */
    static Status validate(const ITensorInfo *bboxes, const ITensorInfo *scores, const ITensorInfo *indices, unsigned int max_output_size,
                           const float score_threshold, const float nms_threshold);
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPP_NONMAXIMUMSUPPRESSION_LAYER_H */
