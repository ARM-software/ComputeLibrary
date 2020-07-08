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
#ifndef ARM_COMPUTE_CPP_NONMAXIMUMSUPPRESSIONKERNEL_LAYER_H
#define ARM_COMPUTE_CPP_NONMAXIMUMSUPPRESSIONKERNEL_LAYER_H

#include "arm_compute/runtime/CPP/ICPPSimpleFunction.h"

#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ITensor;

/** CPP Function to perform non maximum suppression on the bounding boxes and scores
 *
 */
class CPPNonMaximumSuppressionKernel : public ICPPKernel
{
public:
    const char *name() const override
    {
        return "CPPNonMaximumSuppressionKernel";
    }
    /** Default constructor */
    CPPNonMaximumSuppressionKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CPPNonMaximumSuppressionKernel(const CPPNonMaximumSuppressionKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CPPNonMaximumSuppressionKernel &operator=(const CPPNonMaximumSuppressionKernel &) = delete;
    /** Allow instances of this class to be moved */
    CPPNonMaximumSuppressionKernel(CPPNonMaximumSuppressionKernel &&) = default;
    /** Allow instances of this class to be moved */
    CPPNonMaximumSuppressionKernel &operator=(CPPNonMaximumSuppressionKernel &&) = default;
    /** Default destructor */
    ~CPPNonMaximumSuppressionKernel() = default;

    /** Configure the kernel to perform non maximal suppression
     *
     * @param[in]  input_bboxes    The input bounding boxes. Data types supported: F32.
     * @param[in]  input_scores    The corresponding input confidence. Same as @p input_bboxes.
     * @param[out] output_indices  The kept indices of bboxes after nms. Data types supported: S32.
     * @param[in]  max_output_size An integer tensor representing the maximum number of boxes to be selected by non max suppression.
     * @param[in]  score_threshold The threshold used to filter detection results.
     * @param[in]  iou_threshold   The threshold used in non maximum suppression.
     *
     */
    void configure(const ITensor *input_bboxes, const ITensor *input_scores, ITensor *output_indices, unsigned int max_output_size, const float score_threshold, const float iou_threshold);

    /** Static function to check if given arguments will lead to a valid configuration of @ref CPPNonMaximumSuppressionKernel
     *
     * @param[in]  input_bboxes    The input bounding boxes tensor info. Data types supported: F32.
     * @param[in]  input_scores    The corresponding input confidence tensor info. Same as @p input_bboxes.
     * @param[out] output_indices  The kept indices of bboxes after nms tensor info. Data types supported: S32.
     * @param[in]  max_output_size An integer tensor representing the maximum number of boxes to be selected by non max suppression.
     * @param[in]  score_threshold The threshold used to filter detection results.
     * @param[in]  iou_threshold   The threshold used in non maximum suppression.
     *
     */
    static Status validate(const ITensorInfo *input_bboxes, const ITensorInfo *input_scores, const ITensorInfo *output_indices, unsigned int max_output_size,
                           const float score_threshold, const float iou_threshold);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    const ITensor *_input_bboxes;
    const ITensor *_input_scores;
    ITensor       *_output_indices;
    unsigned int   _max_output_size;
    float          _score_threshold;
    float          _iou_threshold;

    unsigned int _num_boxes;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPP_NONMAXIMUMSUPPRESSIONKERNEL_LAYER_H */
