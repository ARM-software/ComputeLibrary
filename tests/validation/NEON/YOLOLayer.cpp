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
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEYOLOLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ActivationFunctionsDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/YOLOLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Tolerance */
constexpr AbsoluteTolerance<float> tolerance_f32(1e-6f);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
constexpr RelativeTolerance<float> tolerance_f16(0.01f);
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

/** Floating point data sets. */
const auto YOLODataset = combine(combine(combine(combine(framework::dataset::make("InPlace", { false, true }), framework::dataset::make("ActivationFunction",
                                                         ActivationLayerInfo::ActivationFunction::LOGISTIC)),
                                                 framework::dataset::make("AlphaBeta", { 0.5f, 1.f })),
                                         framework::dataset::make("Classes", 40)),
                                 framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC }));
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(YOLOLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(16U, 16U, 6U), 1, DataType::U8),  // Wrong input data type
                                                       TensorInfo(TensorShape(16U, 16U, 6U), 1, DataType::F32),  // Invalid activation info
                                                       TensorInfo(TensorShape(16U, 16U, 6U), 1, DataType::F32),  // Wrong output data type
                                                       TensorInfo(TensorShape(16U, 16U, 6U), 1, DataType::F32),  // wrong number of classes
                                                       TensorInfo(TensorShape(16U, 16U, 6U), 1, DataType::F32),  // Mismatching shapes
                                                       TensorInfo(TensorShape(17U, 16U, 6U), 1, DataType::F32),  // shrink window
                                                       TensorInfo(TensorShape(17U, 16U, 7U), 1, DataType::F32),  // channels not multiple of (num_classes + 5)
                                                       TensorInfo(TensorShape(16U, 16U, 6U), 1, DataType::F32),  // Valid
                                                     }),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(16U, 16U, 6U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(16U, 16U, 6U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(16U, 16U, 6U), 1, DataType::U16),
                                                       TensorInfo(TensorShape(16U, 16U, 6U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(16U, 11U, 6U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(16U, 16U, 6U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(16U, 16U, 7U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(16U, 16U, 6U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("ActivationInfo", { ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC),
                                                            ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC),
                                                     })),
               framework::dataset::make("Numclasses", { 1, 1, 1, 0, 1, 1, 1, 1
                                                     })),
               framework::dataset::make("Expected", { false, false, false, false, false, false, false, true})),
               input_info, output_info, act_info, num_classes, expected)
{
    ARM_COMPUTE_EXPECT(bool(NEYOLOLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), act_info, num_classes)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEYOLOLayerFixture = YOLOValidationFixture<Tensor, Accessor, NEYOLOLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEYOLOLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallYOLOShapes(), YOLODataset), framework::dataset::make("DataType",
                                                                                                       DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEYOLOLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeYOLOShapes(), YOLODataset), framework::dataset::make("DataType",
                                                                                                     DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // FP32

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEYOLOLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallYOLOShapes(), YOLODataset), framework::dataset::make("DataType",
                                                                                                      DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEYOLOLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeYOLOShapes(), YOLODataset), framework::dataset::make("DataType",
                                                                                                    DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END() // FP16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
TEST_SUITE_END() // Float

TEST_SUITE_END() // YOLOLayer
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
