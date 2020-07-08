/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CPP_DETECTION_POSTPROCESS_H
#define ARM_COMPUTE_CPP_DETECTION_POSTPROCESS_H

#include "arm_compute/runtime/CPP/ICPPSimpleFunction.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CPP/functions/CPPNonMaximumSuppression.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/Tensor.h"

#include <map>

namespace arm_compute
{
class ITensor;

/** CPP Function to generate the detection output based on center size encoded boxes, class prediction and anchors
 *  by doing non maximum suppression.
 *
 * @note Intended for use with MultiBox detection method.
 */
class CPPDetectionPostProcessLayer : public IFunction
{
public:
    /** Constructor */
    CPPDetectionPostProcessLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CPPDetectionPostProcessLayer(const CPPDetectionPostProcessLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CPPDetectionPostProcessLayer &operator=(const CPPDetectionPostProcessLayer &) = delete;
    /** Configure the detection output layer CPP function
     *
     * @param[in]  input_box_encoding The bounding box input tensor. Data types supported: F32/QASYMM8/QASYMM8_SIGNED.
     * @param[in]  input_score        The class prediction input tensor. Data types supported: Same as @p input_box_encoding.
     * @param[in]  input_anchors      The anchors input tensor. Data types supported: Same as @p input_box_encoding.
     * @param[out] output_boxes       The boxes output tensor. Data types supported: F32.
     * @param[out] output_classes     The classes output tensor. Data types supported: Same as @p output_boxes.
     * @param[out] output_scores      The scores output tensor. Data types supported: Same as @p output_boxes.
     * @param[out] num_detection      The number of output detection. Data types supported: Same as @p output_boxes.
     * @param[in]  info               (Optional) DetectionPostProcessLayerInfo information.
     *
     * @note Output contains all the detections. Of those, only the ones selected by the valid region are valid.
     */
    void configure(const ITensor *input_box_encoding, const ITensor *input_score, const ITensor *input_anchors,
                   ITensor *output_boxes, ITensor *output_classes, ITensor *output_scores, ITensor *num_detection, DetectionPostProcessLayerInfo info = DetectionPostProcessLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref CPPDetectionPostProcessLayer
     *
     * @param[in]  input_box_encoding The bounding box input tensor info. Data types supported: F32/QASYMM8/QASYMM8_SIGNED.
     * @param[in]  input_class_score  The class prediction input tensor info. Data types supported: Same as @p input_box_encoding.
     * @param[in]  input_anchors      The anchors input tensor. Data types supported: F32, QASYMM8.
     * @param[out] output_boxes       The output tensor. Data types supported: F32.
     * @param[out] output_classes     The output tensor. Data types supported: Same as @p output_boxes.
     * @param[out] output_scores      The output tensor. Data types supported: Same as @p output_boxes.
     * @param[out] num_detection      The number of output detection. Data types supported: Same as @p output_boxes.
     * @param[in]  info               (Optional) DetectionPostProcessLayerInfo information.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input_box_encoding, const ITensorInfo *input_class_score, const ITensorInfo *input_anchors,
                           ITensorInfo *output_boxes, ITensorInfo *output_classes, ITensorInfo *output_scores, ITensorInfo *num_detection,
                           DetectionPostProcessLayerInfo info = DetectionPostProcessLayerInfo());
    // Inherited methods overridden:
    void run() override;

private:
    MemoryGroup                   _memory_group;
    CPPNonMaximumSuppression      _nms;
    const ITensor                *_input_box_encoding;
    const ITensor                *_input_scores;
    const ITensor                *_input_anchors;
    ITensor                      *_output_boxes;
    ITensor                      *_output_classes;
    ITensor                      *_output_scores;
    ITensor                      *_num_detection;
    DetectionPostProcessLayerInfo _info;

    const unsigned int _kBatchSize   = 1;
    const unsigned int _kNumCoordBox = 4;
    unsigned int       _num_boxes;
    unsigned int       _num_classes_with_background;
    unsigned int       _num_max_detected_boxes;
    bool               _dequantize_scores;

    Tensor         _decoded_boxes;
    Tensor         _decoded_scores;
    Tensor         _selected_indices;
    Tensor         _class_scores;
    const ITensor *_input_scores_to_use;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPP_DETECTION_POSTPROCESS_H */
