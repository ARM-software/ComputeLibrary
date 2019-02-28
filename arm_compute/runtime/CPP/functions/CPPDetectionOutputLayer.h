/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_CPP_DETECTION_OUTPUT_LAYER_H__
#define __ARM_COMPUTE_CPP_DETECTION_OUTPUT_LAYER_H__

#include "arm_compute/runtime/CPP/ICPPSimpleFunction.h"

#include "arm_compute/core/Types.h"

#include <map>

namespace arm_compute
{
class ITensor;

// Normalized Bounding Box [xmin, ymin, xmax, ymax]
using NormalizedBBox = std::array<float, 4>;
// LabelBBox used for map label and bounding box
using LabelBBox = std::map<int, std::vector<NormalizedBBox>>;

/** CPP Function to perform non maximum suppression on the bounding boxes and scores
 *
 */
class CPPNonMaximumSuppression : public IFunction
{
public:
    /** Default constructor */
    CPPNonMaximumSuppression();
    /** Configure the function to perform non maximal suppression
     *
     * @param[in]  bboxes          The input bounding boxes. Data types supported: F32.
     * @param[in]  scores          The corresponding input confidence. Same as @p scores.
     * @param[out] indices         The kept indices of bboxes after nms. Data types supported: S32.
     * @param[in]  max_output_size An integer tensor representing the maximum number of boxes to be selected by non max suppression.
     * @param[in]  score_threshold The threshold used to filter detection results.
     * @param[in]  nms_threshold   The threshold used in non maximum suppression.
     *
     */
    void configure(const ITensor *bboxes, const ITensor *scores, ITensor *indices, unsigned int max_output_size, const float score_threshold, const float nms_threshold);

    /** Static function to check if given arguments will lead to a valid configuration of @ref CPPNonMaximumSuppression
     *
     * @param[in]  bboxes          The input bounding boxes. Data types supported: F32.
     * @param[in]  scores          The corresponding input confidence. Same as @p scores.
     * @param[out] indices         The kept indices of bboxes after nms. Data types supported: S32.
     * @param[in]  max_output_size An integer tensor representing the maximum number of boxes to be selected by non max suppression.
     * @param[in]  score_threshold The threshold used to filter detection results.
     * @param[in]  nms_threshold   The threshold used in non maximum suppression.
     *
     */
    static Status validate(const ITensorInfo *bboxes, const ITensorInfo *scores, const ITensorInfo *indices, unsigned int max_output_size,
                           const float score_threshold, const float nms_threshold);

    // Inherited methods overridden:
    void run() override;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CPPNonMaximumSuppression(const CPPNonMaximumSuppression &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CPPNonMaximumSuppression &operator=(const CPPNonMaximumSuppression &) = delete;

private:
    const ITensor *_bboxes;
    const ITensor *_scores;
    ITensor       *_indices;
    unsigned int   _max_output_size;

    float _score_threshold;
    float _nms_threshold;
};

/** CPP Function to generate the detection output based on location and confidence
 * predictions by doing non maximum suppression.
 *
 * @note Intended for use with MultiBox detection method.
 */
class CPPDetectionOutputLayer : public IFunction
{
public:
    /** Default constructor */
    CPPDetectionOutputLayer();
    /** Configure the detection output layer CPP kernel
     *
     * @param[in]  input_loc      The mbox location input tensor of size [C1, N]. Data types supported: F32.
     * @param[in]  input_conf     The mbox confidence input tensor of size [C2, N]. Data types supported: F32.
     * @param[in]  input_priorbox The mbox prior box input tensor of size [C3, 2, N]. Data types supported: F32.
     * @param[out] output         The output tensor of size [7, M]. Data types supported: Same as @p input
     * @param[in]  info           (Optional) DetectionOutputLayerInfo information.
     *
     * @note Output contains all the detections. Of those, only the ones selected by the valid region are valid.
     */
    void configure(const ITensor *input_loc, const ITensor *input_conf, const ITensor *input_priorbox, ITensor *output, DetectionOutputLayerInfo info = DetectionOutputLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref CPPDetectionOutputLayer
     *
     * @param[in] input_loc      The mbox location input tensor info. Data types supported: F32.
     * @param[in] input_conf     The mbox confidence input tensor info. Data types supported: F32.
     * @param[in] input_priorbox The mbox prior box input tensor info. Data types supported: F32.
     * @param[in] output         The output tensor info. Data types supported: Same as @p input
     * @param[in] info           (Optional) DetectionOutputLayerInfo information.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input_loc, const ITensorInfo *input_conf, const ITensorInfo *input_priorbox, const ITensorInfo *output,
                           DetectionOutputLayerInfo info = DetectionOutputLayerInfo());
    // Inherited methods overridden:
    void run() override;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CPPDetectionOutputLayer(const CPPDetectionOutputLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CPPDetectionOutputLayer &operator=(const CPPDetectionOutputLayer &) = delete;

private:
    const ITensor           *_input_loc;
    const ITensor           *_input_conf;
    const ITensor           *_input_priorbox;
    ITensor                 *_output;
    DetectionOutputLayerInfo _info;

    int _num_priors;
    int _num;

    std::vector<LabelBBox> _all_location_predictions;
    std::vector<std::map<int, std::vector<float>>> _all_confidence_scores;
    std::vector<NormalizedBBox> _all_prior_bboxes;
    std::vector<std::array<float, 4>> _all_prior_variances;
    std::vector<LabelBBox> _all_decode_bboxes;
    std::vector<std::map<int, std::vector<int>>> _all_indices;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CPP_DETECTION_OUTPUT_LAYER_H__ */
