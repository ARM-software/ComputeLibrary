/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CPP/functions/CPPDetectionPostProcessLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
template <typename U, typename T>
inline void fill_tensor(U &&tensor, const std::vector<T> &v)
{
    std::memcpy(tensor.data(), v.data(), sizeof(T) * v.size());
}
template <typename U, typename T>
inline void quantize_and_fill_tensor(U &&tensor, const std::vector<T> &v)
{
    QuantizationInfo     qi = tensor.quantization_info();
    std::vector<uint8_t> quantized;
    quantized.reserve(v.size());
    for(auto elem : v)
    {
        quantized.emplace_back(quantize_qasymm8(elem, qi));
    }
    std::memcpy(tensor.data(), quantized.data(), sizeof(uint8_t) * quantized.size());
}
inline QuantizationInfo qinfo_scaleoffset_from_minmax(const float min, const float max)
{
    int           offset = 0;
    float         scale  = 0;
    const uint8_t qmin   = std::numeric_limits<uint8_t>::min();
    const uint8_t qmax   = std::numeric_limits<uint8_t>::max();
    const float   f_qmin = qmin;
    const float   f_qmax = qmax;

    // Continue only if [min,max] is a valid range and not a point
    if(min != max)
    {
        scale                       = (max - min) / (f_qmax - f_qmin);
        const float offset_from_min = f_qmin - min / scale;
        const float offset_from_max = f_qmax - max / scale;

        const float offset_from_min_error = std::abs(f_qmin) + std::abs(min / scale);
        const float offset_from_max_error = std::abs(f_qmax) + std::abs(max / scale);
        const float f_offset              = offset_from_min_error < offset_from_max_error ? offset_from_min : offset_from_max;

        uint8_t uint8_offset = 0;
        if(f_offset < f_qmin)
        {
            uint8_offset = qmin;
        }
        else if(f_offset > f_qmax)
        {
            uint8_offset = qmax;
        }
        else
        {
            uint8_offset = static_cast<uint8_t>(std::round(f_offset));
        }
        offset = uint8_offset;
    }
    return QuantizationInfo(scale, offset);
}

inline void base_test_case(DetectionPostProcessLayerInfo info, DataType data_type, const SimpleTensor<float> &expected_output_boxes,
                           const SimpleTensor<float> &expected_output_classes, const SimpleTensor<float> &expected_output_scores, const SimpleTensor<float> &expected_num_detection,
                           AbsoluteTolerance<float> tolerance_boxes = AbsoluteTolerance<float>(0.1f), AbsoluteTolerance<float> tolerance_others = AbsoluteTolerance<float>(0.1f))
{
    Tensor box_encoding     = create_tensor<Tensor>(TensorShape(4U, 6U, 1U), data_type, 1, qinfo_scaleoffset_from_minmax(-1.0f, 1.0f));
    Tensor class_prediction = create_tensor<Tensor>(TensorShape(3U, 6U, 1U), data_type, 1, qinfo_scaleoffset_from_minmax(0.0f, 1.0f));
    Tensor anchors          = create_tensor<Tensor>(TensorShape(4U, 6U), data_type, 1, qinfo_scaleoffset_from_minmax(0.0f, 100.5f));

    box_encoding.allocator()->allocate();
    class_prediction.allocator()->allocate();
    anchors.allocator()->allocate();

    std::vector<float> box_encoding_vector =
    {
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, -1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    };
    std::vector<float> class_prediction_vector =
    {
        0.0f, 0.7f, 0.68f,
        0.0f, 0.6f, 0.5f,
        0.0f, 0.9f, 0.83f,
        0.0f, 0.91f, 0.97f,
        0.0f, 0.5f, 0.4f,
        0.0f, 0.31f, 0.22f
    };
    std::vector<float> anchors_vector =
    {
        0.4f, 0.4f, 1.1f, 1.1f,
        0.4f, 0.4f, 1.1f, 1.1f,
        0.4f, 0.4f, 1.1f, 1.1f,
        0.4f, 10.4f, 1.1f, 1.1f,
        0.4f, 10.4f, 1.1f, 1.1f,
        0.4f, 100.4f, 1.1f, 1.1f
    };

    // Fill the tensors with random pre-generated values
    if(data_type == DataType::F32)
    {
        fill_tensor(Accessor(box_encoding), box_encoding_vector);
        fill_tensor(Accessor(class_prediction), class_prediction_vector);
        fill_tensor(Accessor(anchors), anchors_vector);
    }
    else
    {
        quantize_and_fill_tensor(Accessor(box_encoding), box_encoding_vector);
        quantize_and_fill_tensor(Accessor(class_prediction), class_prediction_vector);
        quantize_and_fill_tensor(Accessor(anchors), anchors_vector);
    }

    // Determine the output through the CPP kernel
    Tensor                       output_boxes;
    Tensor                       output_classes;
    Tensor                       output_scores;
    Tensor                       num_detection;
    CPPDetectionPostProcessLayer detection;
    detection.configure(&box_encoding, &class_prediction, &anchors, &output_boxes, &output_classes, &output_scores, &num_detection, info);

    output_boxes.allocator()->allocate();
    output_classes.allocator()->allocate();
    output_scores.allocator()->allocate();
    num_detection.allocator()->allocate();

    // Run the kernel
    detection.run();

    // Validate against the expected output
    // Validate output boxes
    validate(Accessor(output_boxes), expected_output_boxes, tolerance_boxes);
    // Validate detection classes
    validate(Accessor(output_classes), expected_output_classes, tolerance_others);
    // Validate detection scores
    validate(Accessor(output_scores), expected_output_scores, tolerance_others);
    // Validate num detections
    validate(Accessor(num_detection), expected_num_detection, tolerance_others);
}
} // namespace

TEST_SUITE(CPP)
TEST_SUITE(DetectionPostProcessLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(zip(zip(
        framework::dataset::make("BoxEncodingsInfo", { TensorInfo(TensorShape(4U, 10U, 1U), 1, DataType::F32),
                                                TensorInfo(TensorShape(4U, 10U, 3U), 1, DataType::F32),  // Mismatching batch_size
                                                TensorInfo(TensorShape(4U, 10U, 1U), 1, DataType::S8), // Unsupported data type
                                                TensorInfo(TensorShape(4U, 10U, 1U), 1, DataType::F32), // Wrong Detection Info
                                                TensorInfo(TensorShape(4U, 10U, 1U), 1, DataType::F32), // Wrong boxes dimensions
                                                TensorInfo(TensorShape(4U, 10U, 1U), 1, DataType::QASYMM8)}), // Wrong score dimension 
        framework::dataset::make("ClassPredsInfo",{ TensorInfo(TensorShape(3U ,10U), 1, DataType::F32),
                                                TensorInfo(TensorShape(3U ,10U), 1, DataType::F32),
                                                TensorInfo(TensorShape(3U ,10U), 1, DataType::F32),
                                                TensorInfo(TensorShape(3U ,10U), 1, DataType::F32),
                                                TensorInfo(TensorShape(3U ,10U), 1, DataType::F32),
                                                TensorInfo(TensorShape(3U ,10U), 1, DataType::QASYMM8)})),
        framework::dataset::make("AnchorsInfo",{ TensorInfo(TensorShape(4U, 10U, 1U), 1, DataType::F32),
                                                TensorInfo(TensorShape(4U, 10U, 1U), 1, DataType::F32),
                                                TensorInfo(TensorShape(4U, 10U, 1U), 1, DataType::F32),
                                                TensorInfo(TensorShape(4U, 10U, 1U), 1, DataType::F32),
                                                TensorInfo(TensorShape(4U, 10U, 1U), 1, DataType::F32),
                                                TensorInfo(TensorShape(4U, 10U, 1U), 1, DataType::QASYMM8)})),
        framework::dataset::make("OutputBoxInfo", { TensorInfo(TensorShape(4U, 3U, 1U), 1, DataType::F32),
                                                TensorInfo(TensorShape(4U, 3U, 1U), 1, DataType::F32),
                                                TensorInfo(TensorShape(4U, 3U, 1U), 1, DataType::S8),
                                                TensorInfo(TensorShape(4U, 3U, 1U), 1, DataType::F32),
                                                TensorInfo(TensorShape(1U, 5U, 1U), 1, DataType::F32),
                                                TensorInfo(TensorShape(4U, 3U, 1U), 1, DataType::F32)})),
        framework::dataset::make("OuputClassesInfo",{ TensorInfo(TensorShape(3U, 1U), 1, DataType::F32),
                                                TensorInfo(TensorShape(3U, 1U), 1, DataType::F32),
                                                TensorInfo(TensorShape(3U, 1U), 1, DataType::F32),
                                                TensorInfo(TensorShape(3U, 1U), 1, DataType::F32),
                                                TensorInfo(TensorShape(3U, 1U), 1, DataType::F32),
                                                TensorInfo(TensorShape(6U, 1U), 1, DataType::F32)})),
        framework::dataset::make("OutputScoresInfo",{ TensorInfo(TensorShape(3U, 1U), 1, DataType::F32),
                                                TensorInfo(TensorShape(3U, 1U), 1, DataType::F32),
                                                TensorInfo(TensorShape(3U, 1U), 1, DataType::F32),
                                                TensorInfo(TensorShape(3U, 1U), 1, DataType::F32),
                                                TensorInfo(TensorShape(3U, 1U), 1, DataType::F32),
                                                TensorInfo(TensorShape(6U, 1U), 1, DataType::F32)})),
        framework::dataset::make("NumDetectionsInfo",{ TensorInfo(TensorShape(1U), 1, DataType::F32),
                                                TensorInfo(TensorShape(1U), 1, DataType::F32),
                                                TensorInfo(TensorShape(1U), 1, DataType::F32),
                                                TensorInfo(TensorShape(1U), 1, DataType::F32),
                                                TensorInfo(TensorShape(1U), 1, DataType::F32),
                                                TensorInfo(TensorShape(1U), 1, DataType::F32)})),
        framework::dataset::make("DetectionPostProcessLayerInfo",{ DetectionPostProcessLayerInfo(3, 1, 0.0f, 0.5f, 2, {0.1f,0.1f,0.1f,0.1f}),
                                                DetectionPostProcessLayerInfo(3, 1, 0.0f, 0.5f, 2, {0.1f,0.1f,0.1f,0.1f}),
                                                DetectionPostProcessLayerInfo(3, 1, 0.0f, 0.5f, 2, {0.1f,0.1f,0.1f,0.1f}),
                                                DetectionPostProcessLayerInfo(3, 1, 0.0f, 1.5f, 2, {0.0f,0.1f,0.1f,0.1f}),
                                                DetectionPostProcessLayerInfo(3, 1, 0.0f, 0.5f, 2, {0.1f,0.1f,0.1f,0.1f}),
                                                DetectionPostProcessLayerInfo(3, 1, 0.0f, 0.5f, 2, {0.1f,0.1f,0.1f,0.1f})})),
        framework::dataset::make("Expected", {true, false, false, false, false, false })),
        box_encodings_info, classes_info, anchors_info, output_boxes_info, output_classes_info,output_scores_info, num_detection_info, detect_info, expected)
{
    const Status status = CPPDetectionPostProcessLayer::validate(&box_encodings_info.clone()->set_is_resizable(false),
            &classes_info.clone()->set_is_resizable(false),
            &anchors_info.clone()->set_is_resizable(false),
            &output_boxes_info.clone()->set_is_resizable(false),
            &output_classes_info.clone()->set_is_resizable(false),
            &output_scores_info.clone()->set_is_resizable(false), &num_detection_info.clone()->set_is_resizable(false), detect_info);
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE(F32)
TEST_CASE(Float_general, framework::DatasetMode::ALL)
{
    DetectionPostProcessLayerInfo info = DetectionPostProcessLayerInfo(3 /*max_detections*/, 1 /*max_classes_per_detection*/, 0.0 /*nms_score_threshold*/,
                                                                       0.5 /*nms_iou_threshold*/, 2 /*num_classes*/, { 11.0, 11.0, 6.0, 6.0 } /*scale*/);
    // Fill expected detection boxes
    SimpleTensor<float> expected_output_boxes(TensorShape(4U, 3U), DataType::F32);
    fill_tensor(expected_output_boxes, std::vector<float> { -0.15, 9.85, 0.95, 10.95, -0.15, -0.15, 0.95, 0.95, -0.15, 99.85, 0.95, 100.95 });
    // Fill expected detection classes
    SimpleTensor<float> expected_output_classes(TensorShape(3U), DataType::F32);
    fill_tensor(expected_output_classes, std::vector<float> { 1.0f, 0.0f, 0.0f });
    // Fill expected detection scores
    SimpleTensor<float> expected_output_scores(TensorShape(3U), DataType::F32);
    fill_tensor(expected_output_scores, std::vector<float> { 0.97f, 0.95f, 0.31f });
    // Fill expected num detections
    SimpleTensor<float> expected_num_detection(TensorShape(1U), DataType::F32);
    fill_tensor(expected_num_detection, std::vector<float> { 3.f });
    // Run base test
    base_test_case(info, DataType::F32, expected_output_boxes, expected_output_classes, expected_output_scores, expected_num_detection);
}

TEST_CASE(Float_fast, framework::DatasetMode::ALL)
{
    DetectionPostProcessLayerInfo info = DetectionPostProcessLayerInfo(3 /*max_detections*/, 1 /*max_classes_per_detection*/, 0.0 /*nms_score_threshold*/,
                                                                       0.5 /*nms_iou_threshold*/, 2 /*num_classes*/, { 11.0, 11.0, 6.0, 6.0 } /*scale*/,
                                                                       false /*use_regular_nms*/, 1 /*detections_per_class*/);

    // Fill expected detection boxes
    SimpleTensor<float> expected_output_boxes(TensorShape(4U, 3U), DataType::F32);
    fill_tensor(expected_output_boxes, std::vector<float> { -0.15, 9.85, 0.95, 10.95, -0.15, -0.15, 0.95, 0.95, -0.15, 99.85, 0.95, 100.95 });
    // Fill expected detection classes
    SimpleTensor<float> expected_output_classes(TensorShape(3U), DataType::F32);
    fill_tensor(expected_output_classes, std::vector<float> { 1.0f, 0.0f, 0.0f });
    // Fill expected detection scores
    SimpleTensor<float> expected_output_scores(TensorShape(3U), DataType::F32);
    fill_tensor(expected_output_scores, std::vector<float> { 0.97f, 0.95f, 0.31f });
    // Fill expected num detections
    SimpleTensor<float> expected_num_detection(TensorShape(1U), DataType::F32);
    fill_tensor(expected_num_detection, std::vector<float> { 3.f });

    // Run base test
    base_test_case(info, DataType::F32, expected_output_boxes, expected_output_classes, expected_output_scores, expected_num_detection);
}

TEST_CASE(Float_regular, framework::DatasetMode::ALL)
{
    DetectionPostProcessLayerInfo info = DetectionPostProcessLayerInfo(3 /*max_detections*/, 1 /*max_classes_per_detection*/, 0.0 /*nms_score_threshold*/,
                                                                       0.5 /*nms_iou_threshold*/, 2 /*num_classes*/, { 11.0, 11.0, 6.0, 6.0 } /*scale*/,
                                                                       true /*use_regular_nms*/, 1 /*detections_per_class*/);

    // Fill expected detection boxes
    SimpleTensor<float> expected_output_boxes(TensorShape(4U, 3U), DataType::F32);
    fill_tensor(expected_output_boxes, std::vector<float> { -0.15, 9.85, 0.95, 10.95, -0.15, 9.85, 0.95, 10.95, 0.0f, 0.0f, 0.0f, 0.0f });
    // Fill expected detection classes
    SimpleTensor<float> expected_output_classes(TensorShape(3U), DataType::F32);
    fill_tensor(expected_output_classes, std::vector<float> { 1.0f, 0.0f, 0.0f });
    // Fill expected detection scores
    SimpleTensor<float> expected_output_scores(TensorShape(3U), DataType::F32);
    fill_tensor(expected_output_scores, std::vector<float> { 0.97f, 0.91f, 0.0f });
    // Fill expected num detections
    SimpleTensor<float> expected_num_detection(TensorShape(1U), DataType::F32);
    fill_tensor(expected_num_detection, std::vector<float> { 2.f });

    // Run test
    base_test_case(info, DataType::F32, expected_output_boxes, expected_output_classes, expected_output_scores, expected_num_detection);
}
TEST_SUITE_END() // F32

TEST_SUITE(QASYMM8)
TEST_CASE(Quantized_general, framework::DatasetMode::ALL)
{
    DetectionPostProcessLayerInfo info = DetectionPostProcessLayerInfo(3 /*max_detections*/, 1 /*max_classes_per_detection*/, 0.0 /*nms_score_threshold*/,
                                                                       0.5 /*nms_iou_threshold*/, 2 /*num_classes*/, { 11.0, 11.0, 6.0, 6.0 } /*scale*/);

    // Fill expected detection boxes
    SimpleTensor<float> expected_output_boxes(TensorShape(4U, 3U), DataType::F32);
    fill_tensor(expected_output_boxes, std::vector<float> { -0.15, 9.85, 0.95, 10.95, -0.15, -0.15, 0.95, 0.95, -0.15, 99.85, 0.95, 100.95 });
    // Fill expected detection classes
    SimpleTensor<float> expected_output_classes(TensorShape(3U), DataType::F32);
    fill_tensor(expected_output_classes, std::vector<float> { 1.0f, 0.0f, 0.0f });
    // Fill expected detection scores
    SimpleTensor<float> expected_output_scores(TensorShape(3U), DataType::F32);
    fill_tensor(expected_output_scores, std::vector<float> { 0.97f, 0.95f, 0.31f });
    // Fill expected num detections
    SimpleTensor<float> expected_num_detection(TensorShape(1U), DataType::F32);
    fill_tensor(expected_num_detection, std::vector<float> { 3.f });
    // Run test
    base_test_case(info, DataType::QASYMM8, expected_output_boxes, expected_output_classes, expected_output_scores, expected_num_detection, AbsoluteTolerance<float>(0.3f));
}

TEST_CASE(Quantized_fast, framework::DatasetMode::ALL)
{
    DetectionPostProcessLayerInfo info = DetectionPostProcessLayerInfo(3 /*max_detections*/, 1 /*max_classes_per_detection*/, 0.0 /*nms_score_threshold*/,
                                                                       0.5 /*nms_iou_threshold*/, 2 /*num_classes*/, { 11.0, 11.0, 6.0, 6.0 } /*scale*/,
                                                                       false /*use_regular_nms*/, 1 /*detections_per_class*/);

    // Fill expected detection boxes
    SimpleTensor<float> expected_output_boxes(TensorShape(4U, 3U), DataType::F32);
    fill_tensor(expected_output_boxes, std::vector<float> { -0.15, 9.85, 0.95, 10.95, -0.15, -0.15, 0.95, 0.95, -0.15, 99.85, 0.95, 100.95 });
    // Fill expected detection classes
    SimpleTensor<float> expected_output_classes(TensorShape(3U), DataType::F32);
    fill_tensor(expected_output_classes, std::vector<float> { 1.0f, 0.0f, 0.0f });
    // Fill expected detection scores
    SimpleTensor<float> expected_output_scores(TensorShape(3U), DataType::F32);
    fill_tensor(expected_output_scores, std::vector<float> { 0.97f, 0.95f, 0.31f });
    // Fill expected num detections
    SimpleTensor<float> expected_num_detection(TensorShape(1U), DataType::F32);
    fill_tensor(expected_num_detection, std::vector<float> { 3.f });

    // Run base test
    base_test_case(info, DataType::QASYMM8, expected_output_boxes, expected_output_classes, expected_output_scores, expected_num_detection, AbsoluteTolerance<float>(0.3f));
}

TEST_CASE(Quantized_regular, framework::DatasetMode::ALL)
{
    DetectionPostProcessLayerInfo info = DetectionPostProcessLayerInfo(3 /*max_detections*/, 1 /*max_classes_per_detection*/, 0.0 /*nms_score_threshold*/,
                                                                       0.5 /*nms_iou_threshold*/, 2 /*num_classes*/, { 11.0, 11.0, 6.0, 6.0 } /*scale*/,
                                                                       true /*use_regular_nms*/, 1 /*detections_per_class*/);
    // Fill expected detection boxes
    SimpleTensor<float> expected_output_boxes(TensorShape(4U, 3U), DataType::F32);
    fill_tensor(expected_output_boxes, std::vector<float> { -0.15, 9.85, 0.95, 10.95, -0.15, 9.85, 0.95, 10.95, 0.0f, 0.0f, 0.0f, 0.0f });
    // Fill expected detection classes
    SimpleTensor<float> expected_output_classes(TensorShape(3U), DataType::F32);
    fill_tensor(expected_output_classes, std::vector<float> { 1.0f, 0.0f, 0.0f });
    // Fill expected detection scores
    SimpleTensor<float> expected_output_scores(TensorShape(3U), DataType::F32);
    fill_tensor(expected_output_scores, std::vector<float> { 0.95f, 0.91f, 0.0f });
    // Fill expected num detections
    SimpleTensor<float> expected_num_detection(TensorShape(1U), DataType::F32);
    fill_tensor(expected_num_detection, std::vector<float> { 2.f });

    // Run test
    base_test_case(info, DataType::QASYMM8, expected_output_boxes, expected_output_classes, expected_output_scores, expected_num_detection, AbsoluteTolerance<float>(0.3f));
}

TEST_SUITE_END() // QASYMM8

TEST_SUITE_END() // DetectionPostProcessLayer
TEST_SUITE_END() // CPP
} // namespace validation
} // namespace test
} // namespace arm_compute
