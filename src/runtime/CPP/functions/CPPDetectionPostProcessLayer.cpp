/*
 * Copyright (c) 2019-2020 ARM Limited.
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
#include "arm_compute/runtime/CPP/functions/CPPDetectionPostProcessLayer.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"

#include <cstddef>
#include <ios>
#include <list>

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input_box_encoding, const ITensorInfo *input_class_score, const ITensorInfo *input_anchors,
                          ITensorInfo *output_boxes, ITensorInfo *output_classes, ITensorInfo *output_scores, ITensorInfo *num_detection,
                          DetectionPostProcessLayerInfo info, const unsigned int kBatchSize, const unsigned int kNumCoordBox)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input_box_encoding, input_class_score, input_anchors);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input_box_encoding, 1, DataType::F32, DataType::QASYMM8, DataType::QASYMM8_SIGNED);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input_box_encoding, input_anchors);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input_box_encoding->num_dimensions() > 3, "The location input tensor shape should be [4, N, kBatchSize].");
    if(input_box_encoding->num_dimensions() > 2)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG_VAR(input_box_encoding->dimension(2) != kBatchSize, "The third dimension of the input box_encoding tensor should be equal to %d.", kBatchSize);
    }
    ARM_COMPUTE_RETURN_ERROR_ON_MSG_VAR(input_box_encoding->dimension(0) != kNumCoordBox, "The first dimension of the input box_encoding tensor should be equal to %d.", kNumCoordBox);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input_class_score->dimension(0) != (info.num_classes() + 1),
                                    "The first dimension of the input class_prediction should be equal to the number of classes plus one.");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input_anchors->num_dimensions() > 3, "The anchors input tensor shape should be [4, N, kBatchSize].");
    if(input_anchors->num_dimensions() > 2)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG_VAR(input_anchors->dimension(0) != kNumCoordBox, "The first dimension of the input anchors tensor should be equal to %d.", kNumCoordBox);
    }
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((input_box_encoding->dimension(1) != input_class_score->dimension(1))
                                    || (input_box_encoding->dimension(1) != input_anchors->dimension(1)),
                                    "The second dimension of the inputs should be the same.");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(num_detection->num_dimensions() > 1, "The num_detection output tensor shape should be [M].");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((info.iou_threshold() <= 0.0f) || (info.iou_threshold() > 1.0f), "The intersection over union should be positive and less than 1.");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(info.max_classes_per_detection() <= 0, "The number of max classes per detection should be positive.");

    const unsigned int num_detected_boxes = info.max_detections() * info.max_classes_per_detection();

    // Validate configured outputs
    if(output_boxes->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output_boxes->tensor_shape(), TensorShape(4U, num_detected_boxes, 1U));
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output_boxes, 1, DataType::F32);
    }
    if(output_classes->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output_classes->tensor_shape(), TensorShape(num_detected_boxes, 1U));
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output_classes, 1, DataType::F32);
    }
    if(output_scores->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output_scores->tensor_shape(), TensorShape(num_detected_boxes, 1U));
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output_scores, 1, DataType::F32);
    }
    if(num_detection->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(num_detection->tensor_shape(), TensorShape(1U));
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(num_detection, 1, DataType::F32);
    }

    return Status{};
}

inline void DecodeBoxCorner(BBox &box_centersize, BBox &anchor, Iterator &decoded_it, DetectionPostProcessLayerInfo info)
{
    const float half_factor = 0.5f;

    // BBox is equavalent to CenterSizeEncoding [y,x,h,w]
    const float y_center = box_centersize[0] / info.scale_value_y() * anchor[2] + anchor[0];
    const float x_center = box_centersize[1] / info.scale_value_x() * anchor[3] + anchor[1];
    const float half_h   = half_factor * static_cast<float>(std::exp(box_centersize[2] / info.scale_value_h())) * anchor[2];
    const float half_w   = half_factor * static_cast<float>(std::exp(box_centersize[3] / info.scale_value_w())) * anchor[3];

    // Box Corner encoding boxes are saved as [xmin, ymin, xmax, ymax]
    auto decoded_ptr   = reinterpret_cast<float *>(decoded_it.ptr());
    *(decoded_ptr)     = x_center - half_w; // xmin
    *(1 + decoded_ptr) = y_center - half_h; // ymin
    *(2 + decoded_ptr) = x_center + half_w; // xmax
    *(3 + decoded_ptr) = y_center + half_h; // ymax
}

/** Decode a bbox according to a anchors and scale info.
 *
 * @param[in]  input_box_encoding The input prior bounding boxes.
 * @param[in]  input_anchors      The corresponding input variance.
 * @param[in]  info               The detection informations
 * @param[out] decoded_boxes      The decoded bboxes.
 */
void DecodeCenterSizeBoxes(const ITensor *input_box_encoding, const ITensor *input_anchors, DetectionPostProcessLayerInfo info, Tensor *decoded_boxes)
{
    const QuantizationInfo &qi_box     = input_box_encoding->info()->quantization_info();
    const QuantizationInfo &qi_anchors = input_anchors->info()->quantization_info();
    BBox                    box_centersize{ {} };
    BBox                    anchor{ {} };

    Window win;
    win.use_tensor_dimensions(input_box_encoding->info()->tensor_shape());
    win.set_dimension_step(0U, 4U);
    win.set_dimension_step(1U, 1U);
    Iterator box_it(input_box_encoding, win);
    Iterator anchor_it(input_anchors, win);
    Iterator decoded_it(decoded_boxes, win);

    if(input_box_encoding->info()->data_type() == DataType::QASYMM8)
    {
        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto box_ptr    = reinterpret_cast<const qasymm8_t *>(box_it.ptr());
            const auto anchor_ptr = reinterpret_cast<const qasymm8_t *>(anchor_it.ptr());
            box_centersize        = BBox({ dequantize_qasymm8(*box_ptr, qi_box), dequantize_qasymm8(*(box_ptr + 1), qi_box),
                                           dequantize_qasymm8(*(2 + box_ptr), qi_box), dequantize_qasymm8(*(3 + box_ptr), qi_box)
                                         });
            anchor = BBox({ dequantize_qasymm8(*anchor_ptr, qi_anchors), dequantize_qasymm8(*(anchor_ptr + 1), qi_anchors),
                            dequantize_qasymm8(*(2 + anchor_ptr), qi_anchors), dequantize_qasymm8(*(3 + anchor_ptr), qi_anchors)
                          });
            DecodeBoxCorner(box_centersize, anchor, decoded_it, info);
        },
        box_it, anchor_it, decoded_it);
    }
    else if(input_box_encoding->info()->data_type() == DataType::QASYMM8_SIGNED)
    {
        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto box_ptr    = reinterpret_cast<const qasymm8_signed_t *>(box_it.ptr());
            const auto anchor_ptr = reinterpret_cast<const qasymm8_signed_t *>(anchor_it.ptr());
            box_centersize        = BBox({ dequantize_qasymm8_signed(*box_ptr, qi_box), dequantize_qasymm8_signed(*(box_ptr + 1), qi_box),
                                           dequantize_qasymm8_signed(*(2 + box_ptr), qi_box), dequantize_qasymm8_signed(*(3 + box_ptr), qi_box)
                                         });
            anchor = BBox({ dequantize_qasymm8_signed(*anchor_ptr, qi_anchors), dequantize_qasymm8_signed(*(anchor_ptr + 1), qi_anchors),
                            dequantize_qasymm8_signed(*(2 + anchor_ptr), qi_anchors), dequantize_qasymm8_signed(*(3 + anchor_ptr), qi_anchors)
                          });
            DecodeBoxCorner(box_centersize, anchor, decoded_it, info);
        },
        box_it, anchor_it, decoded_it);
    }
    else
    {
        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto box_ptr    = reinterpret_cast<const float *>(box_it.ptr());
            const auto anchor_ptr = reinterpret_cast<const float *>(anchor_it.ptr());
            box_centersize        = BBox({ *box_ptr, *(box_ptr + 1), *(2 + box_ptr), *(3 + box_ptr) });
            anchor                = BBox({ *anchor_ptr, *(anchor_ptr + 1), *(2 + anchor_ptr), *(3 + anchor_ptr) });
            DecodeBoxCorner(box_centersize, anchor, decoded_it, info);
        },
        box_it, anchor_it, decoded_it);
    }
}

void SaveOutputs(const Tensor *decoded_boxes, const std::vector<int> &result_idx_boxes_after_nms, const std::vector<float> &result_scores_after_nms, const std::vector<int> &result_classes_after_nms,
                 std::vector<unsigned int> &sorted_indices, const unsigned int num_output, const unsigned int max_detections, ITensor *output_boxes, ITensor *output_classes, ITensor *output_scores,
                 ITensor *num_detection)
{
    // xmin,ymin,xmax,ymax -> ymin,xmin,ymax,xmax
    unsigned int i = 0;
    for(; i < num_output; ++i)
    {
        const unsigned int box_in_idx = result_idx_boxes_after_nms[sorted_indices[i]];
        *(reinterpret_cast<float *>(output_boxes->ptr_to_element(Coordinates(0, i)))) = *(reinterpret_cast<float *>(decoded_boxes->ptr_to_element(Coordinates(1, box_in_idx))));
        *(reinterpret_cast<float *>(output_boxes->ptr_to_element(Coordinates(1, i)))) = *(reinterpret_cast<float *>(decoded_boxes->ptr_to_element(Coordinates(0, box_in_idx))));
        *(reinterpret_cast<float *>(output_boxes->ptr_to_element(Coordinates(2, i)))) = *(reinterpret_cast<float *>(decoded_boxes->ptr_to_element(Coordinates(3, box_in_idx))));
        *(reinterpret_cast<float *>(output_boxes->ptr_to_element(Coordinates(3, i)))) = *(reinterpret_cast<float *>(decoded_boxes->ptr_to_element(Coordinates(2, box_in_idx))));
        *(reinterpret_cast<float *>(output_classes->ptr_to_element(Coordinates(i)))) = static_cast<float>(result_classes_after_nms[sorted_indices[i]]);
        *(reinterpret_cast<float *>(output_scores->ptr_to_element(Coordinates(i))))  = result_scores_after_nms[sorted_indices[i]];
    }
    for(; i < max_detections; ++i)
    {
        *(reinterpret_cast<float *>(output_boxes->ptr_to_element(Coordinates(1, i)))) = 0.0f;
        *(reinterpret_cast<float *>(output_boxes->ptr_to_element(Coordinates(0, i)))) = 0.0f;
        *(reinterpret_cast<float *>(output_boxes->ptr_to_element(Coordinates(3, i)))) = 0.0f;
        *(reinterpret_cast<float *>(output_boxes->ptr_to_element(Coordinates(2, i)))) = 0.0f;
        *(reinterpret_cast<float *>(output_classes->ptr_to_element(Coordinates(i)))) = 0.0f;
        *(reinterpret_cast<float *>(output_scores->ptr_to_element(Coordinates(i))))  = 0.0f;
    }
    *(reinterpret_cast<float *>(num_detection->ptr_to_element(Coordinates(0)))) = num_output;
}
} // namespace

CPPDetectionPostProcessLayer::CPPDetectionPostProcessLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _nms(), _input_box_encoding(nullptr), _input_scores(nullptr), _input_anchors(nullptr), _output_boxes(nullptr), _output_classes(nullptr),
      _output_scores(nullptr), _num_detection(nullptr), _info(), _num_boxes(), _num_classes_with_background(), _num_max_detected_boxes(), _dequantize_scores(false), _decoded_boxes(), _decoded_scores(),
      _selected_indices(), _class_scores(), _input_scores_to_use(nullptr)
{
}

void CPPDetectionPostProcessLayer::configure(const ITensor *input_box_encoding, const ITensor *input_scores, const ITensor *input_anchors,
                                             ITensor *output_boxes, ITensor *output_classes, ITensor *output_scores, ITensor *num_detection, DetectionPostProcessLayerInfo info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input_box_encoding, input_scores, input_anchors, output_boxes, output_classes, output_scores);
    _num_max_detected_boxes = info.max_detections() * info.max_classes_per_detection();

    auto_init_if_empty(*output_boxes->info(), TensorInfo(TensorShape(_kNumCoordBox, _num_max_detected_boxes, _kBatchSize), 1, DataType::F32));
    auto_init_if_empty(*output_classes->info(), TensorInfo(TensorShape(_num_max_detected_boxes, _kBatchSize), 1, DataType::F32));
    auto_init_if_empty(*output_scores->info(), TensorInfo(TensorShape(_num_max_detected_boxes, _kBatchSize), 1, DataType::F32));
    auto_init_if_empty(*num_detection->info(), TensorInfo(TensorShape(1U), 1, DataType::F32));

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input_box_encoding->info(), input_scores->info(), input_anchors->info(), output_boxes->info(), output_classes->info(), output_scores->info(),
                                                  num_detection->info(),
                                                  info, _kBatchSize, _kNumCoordBox));

    _input_box_encoding          = input_box_encoding;
    _input_scores                = input_scores;
    _input_anchors               = input_anchors;
    _output_boxes                = output_boxes;
    _output_classes              = output_classes;
    _output_scores               = output_scores;
    _num_detection               = num_detection;
    _info                        = info;
    _num_boxes                   = input_box_encoding->info()->dimension(1);
    _num_classes_with_background = _input_scores->info()->dimension(0);
    _dequantize_scores           = (info.dequantize_scores() && is_data_type_quantized(input_box_encoding->info()->data_type()));

    auto_init_if_empty(*_decoded_boxes.info(), TensorInfo(TensorShape(_kNumCoordBox, _input_box_encoding->info()->dimension(1), _kBatchSize), 1, DataType::F32));
    auto_init_if_empty(*_decoded_scores.info(), TensorInfo(TensorShape(_input_scores->info()->dimension(0), _input_scores->info()->dimension(1), _kBatchSize), 1, DataType::F32));
    auto_init_if_empty(*_selected_indices.info(), TensorInfo(TensorShape(info.use_regular_nms() ? info.detection_per_class() : info.max_detections()), 1, DataType::S32));
    const unsigned int num_classes_per_box = std::min(info.max_classes_per_detection(), info.num_classes());
    auto_init_if_empty(*_class_scores.info(), TensorInfo(info.use_regular_nms() ? TensorShape(_num_boxes) : TensorShape(_num_boxes * num_classes_per_box), 1, DataType::F32));

    _input_scores_to_use = _dequantize_scores ? &_decoded_scores : _input_scores;

    // Manage intermediate buffers
    _memory_group.manage(&_decoded_boxes);
    _memory_group.manage(&_decoded_scores);
    _memory_group.manage(&_selected_indices);
    _memory_group.manage(&_class_scores);
    _nms.configure(&_decoded_boxes, &_class_scores, &_selected_indices, info.use_regular_nms() ? info.detection_per_class() : info.max_detections(), info.nms_score_threshold(), info.iou_threshold());

    // Allocate and reserve intermediate tensors and vectors
    _decoded_boxes.allocator()->allocate();
    _decoded_scores.allocator()->allocate();
    _selected_indices.allocator()->allocate();
    _class_scores.allocator()->allocate();
}

Status CPPDetectionPostProcessLayer::validate(const ITensorInfo *input_box_encoding, const ITensorInfo *input_class_score, const ITensorInfo *input_anchors,
                                              ITensorInfo *output_boxes, ITensorInfo *output_classes, ITensorInfo *output_scores, ITensorInfo *num_detection, DetectionPostProcessLayerInfo info)
{
    constexpr unsigned int kBatchSize             = 1;
    constexpr unsigned int kNumCoordBox           = 4;
    const TensorInfo       _decoded_boxes_info    = TensorInfo(TensorShape(kNumCoordBox, input_box_encoding->dimension(1)), 1, DataType::F32);
    const TensorInfo       _decoded_scores_info   = TensorInfo(TensorShape(input_box_encoding->dimension(1)), 1, DataType::F32);
    const TensorInfo       _selected_indices_info = TensorInfo(TensorShape(info.max_detections()), 1, DataType::S32);

    ARM_COMPUTE_RETURN_ON_ERROR(CPPNonMaximumSuppression::validate(&_decoded_boxes_info, &_decoded_scores_info, &_selected_indices_info, info.max_detections(), info.nms_score_threshold(),
                                                                   info.iou_threshold()));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input_box_encoding, input_class_score, input_anchors, output_boxes, output_classes, output_scores, num_detection, info, kBatchSize, kNumCoordBox));

    return Status{};
}

void CPPDetectionPostProcessLayer::run()
{
    const unsigned int num_classes    = _info.num_classes();
    const unsigned int max_detections = _info.max_detections();

    DecodeCenterSizeBoxes(_input_box_encoding, _input_anchors, _info, &_decoded_boxes);

    // Decode scores if necessary
    if(_dequantize_scores)
    {
        if(_input_box_encoding->info()->data_type() == DataType::QASYMM8)
        {
            for(unsigned int idx_c = 0; idx_c < _num_classes_with_background; ++idx_c)
            {
                for(unsigned int idx_b = 0; idx_b < _num_boxes; ++idx_b)
                {
                    *(reinterpret_cast<float *>(_decoded_scores.ptr_to_element(Coordinates(idx_c, idx_b)))) =
                        dequantize_qasymm8(*(reinterpret_cast<qasymm8_t *>(_input_scores->ptr_to_element(Coordinates(idx_c, idx_b)))), _input_scores->info()->quantization_info());
                }
            }
        }
        else if(_input_box_encoding->info()->data_type() == DataType::QASYMM8_SIGNED)
        {
            for(unsigned int idx_c = 0; idx_c < _num_classes_with_background; ++idx_c)
            {
                for(unsigned int idx_b = 0; idx_b < _num_boxes; ++idx_b)
                {
                    *(reinterpret_cast<float *>(_decoded_scores.ptr_to_element(Coordinates(idx_c, idx_b)))) =
                        dequantize_qasymm8_signed(*(reinterpret_cast<qasymm8_signed_t *>(_input_scores->ptr_to_element(Coordinates(idx_c, idx_b)))), _input_scores->info()->quantization_info());
                }
            }
        }
    }

    // Regular NMS
    if(_info.use_regular_nms())
    {
        std::vector<int>          result_idx_boxes_after_nms;
        std::vector<int>          result_classes_after_nms;
        std::vector<float>        result_scores_after_nms;
        std::vector<unsigned int> sorted_indices;

        for(unsigned int c = 0; c < num_classes; ++c)
        {
            // For each boxes get scores of the boxes for the class c
            for(unsigned int i = 0; i < _num_boxes; ++i)
            {
                *(reinterpret_cast<float *>(_class_scores.ptr_to_element(Coordinates(i)))) =
                    *(reinterpret_cast<float *>(_input_scores_to_use->ptr_to_element(Coordinates(c + 1, i)))); // i * _num_classes_with_background + c + 1
            }

            // Run Non-maxima Suppression
            _nms.run();

            for(unsigned int i = 0; i < _info.detection_per_class(); ++i)
            {
                const auto selected_index = *(reinterpret_cast<int *>(_selected_indices.ptr_to_element(Coordinates(i))));
                if(selected_index == -1)
                {
                    // Nms will return -1 for all the last M-elements not valid
                    break;
                }
                result_idx_boxes_after_nms.emplace_back(selected_index);
                result_scores_after_nms.emplace_back((reinterpret_cast<float *>(_class_scores.buffer()))[selected_index]);
                result_classes_after_nms.emplace_back(c);
            }
        }

        // We select the max detection numbers of the highest score of all classes
        const auto num_selected = result_scores_after_nms.size();
        const auto num_output   = std::min<unsigned int>(max_detections, num_selected);

        // Sort selected indices based on result scores
        sorted_indices.resize(num_selected);
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::partial_sort(sorted_indices.data(),
                          sorted_indices.data() + num_output,
                          sorted_indices.data() + num_selected,
                          [&](unsigned int first, unsigned int second)
        {

            return result_scores_after_nms[first] > result_scores_after_nms[second];
        });

        SaveOutputs(&_decoded_boxes, result_idx_boxes_after_nms, result_scores_after_nms, result_classes_after_nms, sorted_indices,
                    num_output, max_detections, _output_boxes, _output_classes, _output_scores, _num_detection);
    }
    // Fast NMS
    else
    {
        const unsigned int num_classes_per_box = std::min<unsigned int>(_info.max_classes_per_detection(), _info.num_classes());
        std::vector<float> max_scores;
        std::vector<int>   box_indices;
        std::vector<int>   max_score_classes;

        for(unsigned int b = 0; b < _num_boxes; ++b)
        {
            std::vector<float> box_scores;
            for(unsigned int c = 0; c < num_classes; ++c)
            {
                box_scores.emplace_back(*(reinterpret_cast<float *>(_input_scores_to_use->ptr_to_element(Coordinates(c + 1, b)))));
            }

            std::vector<unsigned int> max_score_indices;
            max_score_indices.resize(_info.num_classes());
            std::iota(max_score_indices.data(), max_score_indices.data() + _info.num_classes(), 0);
            std::partial_sort(max_score_indices.data(),
                              max_score_indices.data() + num_classes_per_box,
                              max_score_indices.data() + num_classes,
                              [&](unsigned int first, unsigned int second)
            {
                return box_scores[first] > box_scores[second];
            });

            for(unsigned int i = 0; i < num_classes_per_box; ++i)
            {
                const float score_to_add                                                                             = box_scores[max_score_indices[i]];
                *(reinterpret_cast<float *>(_class_scores.ptr_to_element(Coordinates(b * num_classes_per_box + i)))) = score_to_add;
                max_scores.emplace_back(score_to_add);
                box_indices.emplace_back(b);
                max_score_classes.emplace_back(max_score_indices[i]);
            }
        }

        // Run Non-maxima Suppression
        _nms.run();
        std::vector<unsigned int> selected_indices;
        for(unsigned int i = 0; i < max_detections; ++i)
        {
            // NMS returns M valid indices, the not valid tail is filled with -1
            if(*(reinterpret_cast<int *>(_selected_indices.ptr_to_element(Coordinates(i)))) == -1)
            {
                // Nms will return -1 for all the last M-elements not valid
                break;
            }
            selected_indices.emplace_back(*(reinterpret_cast<int *>(_selected_indices.ptr_to_element(Coordinates(i)))));
        }
        // We select the max detection numbers of the highest score of all classes
        const auto num_output = std::min<unsigned int>(_info.max_detections(), selected_indices.size());

        SaveOutputs(&_decoded_boxes, box_indices, max_scores, max_score_classes, selected_indices,
                    num_output, max_detections, _output_boxes, _output_classes, _output_scores, _num_detection);
    }
}
} // namespace arm_compute
