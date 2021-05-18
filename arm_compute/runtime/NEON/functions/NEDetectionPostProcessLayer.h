/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NE_DETECTION_POSTPROCESS_H
#define ARM_COMPUTE_NE_DETECTION_POSTPROCESS_H

#include "arm_compute/runtime/NEON/INESimpleFunction.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CPP/functions/CPPDetectionPostProcessLayer.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/NEON/functions/NEDequantizationLayer.h"
#include "arm_compute/runtime/Tensor.h"

#include <map>

namespace arm_compute
{
class ITensor;

/** NE Function to generate the detection output based on center size encoded boxes, class prediction and anchors
 *  by doing non maximum suppression.
 *
 * @note Intended for use with MultiBox detection method.
 */
class NEDetectionPostProcessLayer : public IFunction
{
public:
    /** Constructor */
    NEDetectionPostProcessLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDetectionPostProcessLayer(const NEDetectionPostProcessLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDetectionPostProcessLayer &operator=(const NEDetectionPostProcessLayer &) = delete;
    /** Default destructor */
    ~NEDetectionPostProcessLayer() = default;
    /** Configure the detection output layer NE function
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0 - src2    |dst0 - dst3    |
     * |:--------------|:--------------|
     * |QASYMM8        |F32            |
     * |QASYMM8_SIGNED |F32            |
     * |F32            |F32            |
     *
     * @param[in]  input_box_encoding The bounding box input tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/F32.
     * @param[in]  input_score        The class prediction input tensor. Data types supported: same as @p input_box_encoding.
     * @param[in]  input_anchors      The anchors input tensor. Data types supported: same as @p input_box_encoding.
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
    /** Static function to check if given info will lead to a valid configuration of @ref NEDetectionPostProcessLayer
     *
     * @param[in] input_box_encoding The bounding box input tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F32.
     * @param[in] input_class_score  The class prediction input tensor info. Data types supported: same as @p input_box_encoding.
     * @param[in] input_anchors      The anchors input tensor info. Data types supported: same as @p input_box_encoding.
     * @param[in] output_boxes       The output tensor info. Data types supported: F32.
     * @param[in] output_classes     The output tensor info. Data types supported: Same as @p output_boxes.
     * @param[in] output_scores      The output tensor info. Data types supported: Same as @p output_boxes.
     * @param[in] num_detection      The number of output detection tensor info. Data types supported: Same as @p output_boxes.
     * @param[in] info               (Optional) DetectionPostProcessLayerInfo information.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input_box_encoding, const ITensorInfo *input_class_score, const ITensorInfo *input_anchors,
                           ITensorInfo *output_boxes, ITensorInfo *output_classes, ITensorInfo *output_scores, ITensorInfo *num_detection,
                           DetectionPostProcessLayerInfo info = DetectionPostProcessLayerInfo());
    // Inherited methods overridden:
    void run() override;

private:
    MemoryGroup _memory_group;

    NEDequantizationLayer        _dequantize;
    CPPDetectionPostProcessLayer _detection_post_process;

    Tensor _decoded_scores;
    bool   _run_dequantize;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NE_DETECTION_POSTPROCESS_H */
