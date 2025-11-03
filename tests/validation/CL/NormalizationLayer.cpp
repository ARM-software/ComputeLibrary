/*
 * Copyright (c) 2017-2021, 2025 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLNormalizationLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/NormalizationTypesDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/NormalizationLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
using framework::dataset::make;
namespace
{
/** Tolerance for float operations */
RelativeTolerance<half>  tolerance_f16(half(0.2));
RelativeTolerance<float> tolerance_f32(0.05f);

/** Input data set. */
const auto NormalizationDatasetFP16 = combine(make("NormType", { NormType::IN_MAP_1D, NormType::IN_MAP_2D, NormType::CROSS_MAP }),
                                                              make("NormalizationSize", 3, 9, 2),
                                                      make("Beta", { 0.5f, 1.f, 2.f }),
                                              make("IsScaled", { true }));

const auto NormalizationDatasetFP32 = combine(make("NormType", { NormType::IN_MAP_1D, NormType::IN_MAP_2D, NormType::CROSS_MAP }),
                                                              make("NormalizationSize", 3, 9, 2),
                                                      make("Beta", { 0.5f, 1.f, 2.f }),
                                              make("IsScaled", { true, false }));
} // namespace

TEST_SUITE(CL)
TEST_SUITE(NormalizationLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Mismatching data type input/output
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Mismatching shapes
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Even normalization
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Windows shrinking for NCHW
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                     }),
               make("OutputInfo",{ TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(27U, 11U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                     }),
               make("NormInfo",  { NormalizationLayerInfo(NormType::IN_MAP_1D, 5),
                                                       NormalizationLayerInfo(NormType::IN_MAP_1D, 5),
                                                       NormalizationLayerInfo(NormType::IN_MAP_1D, 4),
                                                       NormalizationLayerInfo(NormType::IN_MAP_2D, 5),
                                                       NormalizationLayerInfo(NormType::CROSS_MAP, 5),
                                                      }),
               make("Expected", { false, false, false, false, true })),
               input_info, output_info, norm_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLNormalizationLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), norm_info)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLNormalizationLayerFixture = NormalizationValidationFixture<CLTensor, CLAccessor, CLNormalizationLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLNormalizationLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallShapes(), NormalizationDatasetFP16,
                                                                                                                       make("DataType", DataType::F16),
                                                                                                               make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLNormalizationLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), NormalizationDatasetFP16,
                                                                                                                     make("DataType", DataType::F16),
                                                                                                             make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END() // FP16

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLNormalizationLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallShapes(), NormalizationDatasetFP32,
                                                                                                                        make("DataType", DataType::F32),
                                                                                                                make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLNormalizationLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), NormalizationDatasetFP32,
                                                                                                                      make("DataType", DataType::F32),
                                                                                                              make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

TEST_SUITE_END() // NormalizationLayer
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
