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
#include "arm_compute/runtime/NEON/functions/NEDetectionPostProcessLayer.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"

#include <cstddef>
#include <ios>
#include <list>

namespace arm_compute
{
NEDetectionPostProcessLayer::NEDetectionPostProcessLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _dequantize(), _detection_post_process(), _decoded_scores(), _run_dequantize(false)
{
}

void NEDetectionPostProcessLayer::configure(const ITensor *input_box_encoding, const ITensor *input_scores, const ITensor *input_anchors,
                                            ITensor *output_boxes, ITensor *output_classes, ITensor *output_scores, ITensor *num_detection, DetectionPostProcessLayerInfo info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input_box_encoding, input_scores, input_anchors, output_boxes, output_classes, output_scores);
    ARM_COMPUTE_ERROR_THROW_ON(NEDetectionPostProcessLayer::validate(input_box_encoding->info(), input_scores->info(), input_anchors->info(), output_boxes->info(), output_classes->info(),
                                                                     output_scores->info(),
                                                                     num_detection->info(), info));

    const ITensor                *input_scores_to_use = input_scores;
    DetectionPostProcessLayerInfo info_to_use         = info;
    _run_dequantize                                   = is_data_type_quantized(input_box_encoding->info()->data_type());

    if(_run_dequantize)
    {
        _memory_group.manage(&_decoded_scores);

        _dequantize.configure(input_scores, &_decoded_scores);

        input_scores_to_use = &_decoded_scores;

        // Create a new info struct to avoid dequantizing in the CPP layer
        std::array<float, 4> scales_values{ info.scale_value_y(), info.scale_value_x(), info.scale_value_h(), info.scale_value_w() };
        DetectionPostProcessLayerInfo info_quantized(info.max_detections(), info.max_classes_per_detection(), info.nms_score_threshold(), info.iou_threshold(), info.num_classes(),
                                                     scales_values, info.use_regular_nms(), info.detection_per_class(), false);
        info_to_use = info_quantized;
    }

    _detection_post_process.configure(input_box_encoding, input_scores_to_use, input_anchors, output_boxes, output_classes, output_scores, num_detection, info_to_use);
    _decoded_scores.allocator()->allocate();
}

Status NEDetectionPostProcessLayer::validate(const ITensorInfo *input_box_encoding, const ITensorInfo *input_scores, const ITensorInfo *input_anchors,
                                             ITensorInfo *output_boxes, ITensorInfo *output_classes, ITensorInfo *output_scores, ITensorInfo *num_detection, DetectionPostProcessLayerInfo info)
{
    bool run_dequantize = is_data_type_quantized(input_box_encoding->data_type());
    if(run_dequantize)
    {
        TensorInfo decoded_classes_info = input_scores->clone()->set_is_resizable(true).set_data_type(DataType::F32);
        ARM_COMPUTE_RETURN_ON_ERROR(NEDequantizationLayer::validate(input_scores, &decoded_classes_info));
    }
    ARM_COMPUTE_RETURN_ON_ERROR(CPPDetectionPostProcessLayer::validate(input_box_encoding, input_scores, input_anchors, output_boxes, output_classes, output_scores, num_detection, info));

    return Status{};
}

void NEDetectionPostProcessLayer::run()
{
    MemoryGroupResourceScope scope_mg(_memory_group);

    // Decode scores if necessary
    if(_run_dequantize)
    {
        _dequantize.run();
    }
    _detection_post_process.run();
}
} // namespace arm_compute
