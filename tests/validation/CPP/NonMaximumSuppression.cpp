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
#include "arm_compute/runtime/CPP/functions/CPPNonMaximumSuppression.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/NonMaxSuppressionFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
const auto max_output_boxes_dataset  = framework::dataset::make("MaxOutputBoxes", 1, 10);
const auto score_threshold_dataset   = framework::dataset::make("ScoreThreshold", { 0.1f, 0.5f, 0.f, 1.f });
const auto iou_nms_threshold_dataset = framework::dataset::make("NMSThreshold", { 0.1f, 0.5f, 0.f, 1.f });
const auto NMSParametersSmall        = datasets::Small2DNonMaxSuppressionShapes() * max_output_boxes_dataset * score_threshold_dataset * iou_nms_threshold_dataset;
const auto NMSParametersBig          = datasets::Large2DNonMaxSuppressionShapes() * max_output_boxes_dataset * score_threshold_dataset * iou_nms_threshold_dataset;

} // namespace

TEST_SUITE(CPP)
TEST_SUITE(NMS)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(
                                                framework::dataset::make("BoundingBox",{
                                                                                        TensorInfo(TensorShape(4U, 100U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(1U, 4U, 2U), 1, DataType::F32),    // invalid shape
                                                                                        TensorInfo(TensorShape(4U, 2U), 1, DataType::S32),    // invalid data type
                                                                                        TensorInfo(TensorShape(4U, 3U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(4U, 66U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(4U, 100U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(4U, 100U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(4U, 100U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(4U, 100U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(4U, 100U), 1, DataType::F32),
                                                                                    }),
                                                framework::dataset::make("Scores", {
                                                                                        TensorInfo(TensorShape(100U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(37U, 2U, 13U, 27U), 1, DataType::F32), // invalid shape
                                                                                        TensorInfo(TensorShape(4U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(3U), 1, DataType::U8),  // invalid data type
                                                                                        TensorInfo(TensorShape(66U), 1, DataType::F32),  // invalid data type
                                                                                        TensorInfo(TensorShape(100U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(100U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(100U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(100U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(100U), 1, DataType::F32),
                                                                                    })),
                                                framework::dataset::make("Indices", {
                                                                                        TensorInfo(TensorShape(100U), 1, DataType::S32),
                                                                                        TensorInfo(TensorShape(100U), 1, DataType::S32),
                                                                                        TensorInfo(TensorShape(4U), 1, DataType::S32),
                                                                                        TensorInfo(TensorShape(3U), 1, DataType::S32),
                                                                                        TensorInfo(TensorShape(200U), 1, DataType::S32), // indices bigger than max bbs, OK because max_output is 66
                                                                                        TensorInfo(TensorShape(100U), 1, DataType::F32), // invalid data type
                                                                                        TensorInfo(TensorShape(100U), 1, DataType::S32),
                                                                                        TensorInfo(TensorShape(100U), 1, DataType::S32),
                                                                                        TensorInfo(TensorShape(100U), 1, DataType::S32),
                                                                                        TensorInfo(TensorShape(100U), 1, DataType::S32),

                                                                                    })),
                                                framework::dataset::make("max_output", {
                                                                                        10U, 2U,4U, 3U,66U, 1U,
                                                                                        0U, /* invalid, must be greater than 0 */
                                                                                        10000U, /* OK, clamped to indices' size */
                                                                                        100U,
                                                                                        10U,
                                                                                     })),
                                                framework::dataset::make("score_threshold", {
                                                                                        0.1f, 0.4f, 0.2f,0.8f,0.3f, 0.01f, 0.5f, 0.45f,
                                                                                        -1.f, /* invalid value, must be in [0,1] */
                                                                                        0.5f,
                                                                                     })),
                                                framework::dataset::make("nms_threshold", {
                                                                                        0.3f, 0.7f, 0.1f,0.13f,0.2f, 0.97f, 0.76f, 0.87f, 0.1f,
                                                                                        10.f, /* invalid value, must be in [0,1]*/
                                                                                     })),
                                                framework::dataset::make("Expected", {
                                                                                        true, false, false, false, true, false, false,true, false, false
                                                                                     })),

                                            bbox_info, scores_info, indices_info, max_out, score_threshold, nms_threshold, expected)
{
    ARM_COMPUTE_EXPECT(bool(CPPNonMaximumSuppression::validate(&bbox_info.clone()->set_is_resizable(false),
                                                               &scores_info.clone()->set_is_resizable(false),
                                                               &indices_info.clone()->set_is_resizable(false),
                                max_out,score_threshold,nms_threshold)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

using CPPNonMaxSuppressionFixture = NMSValidationFixture<Tensor, Accessor, CPPNonMaximumSuppression>;

FIXTURE_DATA_TEST_CASE(RunSmall, CPPNonMaxSuppressionFixture, framework::DatasetMode::PRECOMMIT, NMSParametersSmall)
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CPPNonMaxSuppressionFixture, framework::DatasetMode::NIGHTLY, NMSParametersBig)
{
    // Validate output
    validate(Accessor(_target), _reference);
}

TEST_SUITE_END() // NMS
TEST_SUITE_END() // CPP
} // namespace validation
} // namespace test
} // namespace arm_compute
