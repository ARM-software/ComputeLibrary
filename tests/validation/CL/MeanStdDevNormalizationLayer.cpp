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
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLMeanStdDevNormalizationLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/NormalizationTypesDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/MeanStdDevNormalizationLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Tolerance for float operations */
RelativeTolerance<half>  tolerance_f16(half(0.2f));
RelativeTolerance<float> tolerance_f32(1e-8f);
} // namespace

TEST_SUITE(CL)
TEST_SUITE(MeanStdDevNormalizationLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 13U), 1, DataType::F32), // Mismatching data type input/output
                                                       TensorInfo(TensorShape(27U, 13U), 1, DataType::F32), // Mismatching shapes
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),
                                                     }),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(27U, 13U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(27U, 11U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("Expected", { false, false, true })),
               input_info, output_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLMeanStdDevNormalizationLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false))) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLMeanStdDevNormalizationLayerFixture = MeanStdDevNormalizationLayerValidationFixture<CLTensor, CLAccessor, CLMeanStdDevNormalizationLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLMeanStdDevNormalizationLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::Small2DShapes(),
                       framework::dataset::make("DataType", DataType::F16)),
                       framework::dataset::make("InPlace", { false, true })),
                       framework::dataset::make("Epsilon", { 1e-8 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLMeanStdDevNormalizationLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::Large2DShapes(),
                                                                                                                       framework::dataset::make("DataType", DataType::F16)),
                                                                                                                       framework::dataset::make("InPlace", { false, true })),
                                                                                                                       framework::dataset::make("Epsilon", { 1e-8 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END() // FP16

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLMeanStdDevNormalizationLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::Small2DShapes(),
                       framework::dataset::make("DataType", DataType::F32)),
                       framework::dataset::make("InPlace", { false, true })),
                       framework::dataset::make("Epsilon", { 1e-8 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLMeanStdDevNormalizationLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::Large2DShapes(),
                                                                                                                        framework::dataset::make("DataType", DataType::F32)),
                                                                                                                        framework::dataset::make("InPlace", { false, true })),
                                                                                                                        framework::dataset::make("Epsilon", { 1e-8 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

TEST_SUITE_END() // MeanStdNormalizationLayer
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
