/*
 * Copyright (c) 2022 Arm Limited.
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

#include "arm_compute/core/TensorShape.h"
#include "tests/framework/datasets/Datasets.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLPooling3dLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/Pooling3dLayerDataset.h"
#include "tests/datasets/PoolingTypesDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/Pooling3dLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Input data sets for floating-point data types */
const auto Pooling3dLayerDatasetFP = combine(combine(combine(combine(datasets::PoolingTypes(), framework::dataset::make("PoolingSize", { Size3D(2, 3, 2) })),
                                                             framework::dataset::make("Stride", { Size3D(1, 1, 1), Size3D(2, 1, 1), Size3D(1, 2, 1), Size3D(2, 2, 1) })),
                                                     framework::dataset::make("Padding", { Padding3D(0, 1, 0), Padding3D(1, 1, 1) })),
                                             framework::dataset::make("ExcludePadding", { true, false }));

const auto Pooling3dLayerDatasetFPSmall = combine(combine(combine(combine(datasets::PoolingTypes(), framework::dataset::make("PoolingSize", { Size3D(2, 2, 2), Size3D(3, 3, 3) })),
                                                                  framework::dataset::make("Stride", { Size3D(2, 2, 2), Size3D(2, 1, 1) })),
                                                          framework::dataset::make("Padding", { Padding3D(0, 0, 0), Padding3D(1, 1, 1), Padding3D(1, 0, 0) })),
                                                  framework::dataset::make("ExcludePadding", { true, false }));

using ShapeDataset = framework::dataset::ContainerDataset<std::vector<TensorShape>>;

constexpr AbsoluteTolerance<float> tolerance_f32(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for 32-bit floating-point type */
constexpr AbsoluteTolerance<float> tolerance_f16(0.1f);  /**< Tolerance value for comparing reference's output against implementation's output for 16-bit floating-point type */

} // namespace

TEST_SUITE(CL)
TEST_SUITE(Pooling3dLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(2U, 27U, 13U, 4U, 3U), 1, DataType::F32, DataLayout::NDHWC), // Mismatching data type
                                                       TensorInfo(TensorShape(2U, 27U, 13U, 4U, 2U), 1, DataType::F32, DataLayout::NDHWC), // Invalid pad/size combination
                                                       TensorInfo(TensorShape(2U, 27U, 13U, 4U, 2U), 1, DataType::F32, DataLayout::NDHWC), // Invalid pad/size combination
                                                       TensorInfo(TensorShape(2U, 27U, 13U, 4U, 3U), 1, DataType::F32, DataLayout::NDHWC), // Invalid output shape
                                                       TensorInfo(TensorShape(5U, 13U, 15U, 2U, 3U), 1, DataType::F32, DataLayout::NDHWC), // Global Pooling
                                                       TensorInfo(TensorShape(13U,13U, 5U, 1U, 2U), 1, DataType::F32, DataLayout::NDHWC),  // Invalid output Global Pooling
                                                       TensorInfo(TensorShape(5U, 13U, 13U, 4U, 4U), 1, DataType::F32, DataLayout::NDHWC), // Invalid data type
                                                       TensorInfo(TensorShape(5U, 13U, 13U, 4U, 4U), 1, DataType::F32, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(5U, 13U, 13U, 5U, 4U), 1, DataType::F32, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(1U, 16U,  1U, 3U, 4U), 1, DataType::F32, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(5U, 13U, 13U, 4U, 3U), 1, DataType::F32, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(5U, 13U, 13U, 4U, 2U), 1, DataType::F32, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(5U, 13U, 13U, 4U, 3U), 1, DataType::F32, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(5U, 13U, 13U, 4U, 3U), 1, DataType::F32, DataLayout::NDHWC),
                                                     }),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(2U, 25U, 11U, 3U, 3U), 1, DataType::F16, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(2U, 30U, 11U, 3U, 2U), 1, DataType::F32, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(2U, 25U, 16U, 3U, 2U), 1, DataType::F32, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(2U, 27U, 13U, 3U, 3U), 1, DataType::F32, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(5U,  1U,  1U, 1U, 3U), 1, DataType::F32, DataLayout::NDHWC), // Global pooling applied
                                                       TensorInfo(TensorShape(5U,  2U,  2U, 2U, 2U), 1, DataType::F32, DataLayout::NDHWC), // Invalid output Global Pooling
                                                       TensorInfo(TensorShape(5U, 12U, 12U, 3U, 4U), 1, DataType::F32, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(5U, 12U, 12U, 3U, 4U), 1, DataType::QASYMM8, DataLayout::NDHWC), // Invalid data type
                                                       TensorInfo(TensorShape(5U,  1U, 1U, 1U, 4U), 1, DataType::F32, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(1U, 15U, 1U, 2U, 4U), 1, DataType::F32, DataLayout::NDHWC), // Output width larger than input
                                                       TensorInfo(TensorShape(5U, 6U, 6U, 2U, 3U),  1, DataType::F32, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(5U, 6U, 6U, 2U, 2U),  1, DataType::F32, DataLayout::NDHWC), 
                                                       TensorInfo(TensorShape(5U, 6U, 6U, 2U, 3U),  1, DataType::F32, DataLayout::NDHWC),
                                                       TensorInfo(TensorShape(5U, 6U, 6U, 2U, 3U),  1, DataType::F32, DataLayout::NDHWC),
                                                     })),
               framework::dataset::make("PoolInfo",  { Pooling3dLayerInfo(PoolingType::AVG, 3, Size3D(1, 1, 1), Padding3D(0, 0, 0)),
                                                       Pooling3dLayerInfo(PoolingType::AVG, 2, Size3D(1, 1, 1), Padding3D(2, 0, 0)),
                                                       Pooling3dLayerInfo(PoolingType::AVG, 2, Size3D(1, 1, 1), Padding3D(0, 0, 0)),
                                                       Pooling3dLayerInfo(PoolingType::L2,  3, Size3D(1, 1, 1), Padding3D(0, 0, 0)),
                                                       Pooling3dLayerInfo(PoolingType::AVG),
                                                       Pooling3dLayerInfo(PoolingType::MAX),
                                                       Pooling3dLayerInfo(PoolingType::AVG, 2, Size3D(), Padding3D(), false),
                                                       Pooling3dLayerInfo(PoolingType::AVG, 2, Size3D(1U, 1U, 1U), Padding3D(), false),
                                                       Pooling3dLayerInfo(PoolingType::AVG),
                                                       Pooling3dLayerInfo(PoolingType::MAX, 2, Size3D(1, 1, 2), Padding3D(0, 0, 0), false),
                                                       Pooling3dLayerInfo(PoolingType::AVG, 2, Size3D(2U, 2U, 2U), Padding3D(), false),
                                                       Pooling3dLayerInfo(PoolingType::AVG, 1, Size3D(2U, 2U, 2U), Padding3D(2, 2, 2), true), // Pool size is smaller than the padding size with padding excluded
                                                       Pooling3dLayerInfo(PoolingType::AVG, 1, Size3D(2U, 2U, 2U), Padding3D(2, 2, 2), false), // Pool size is smaller than the padding size with padding included
                                                       Pooling3dLayerInfo(PoolingType::AVG, 3, Size3D(2U, 2U, 2U), Padding3D(2,1,2,2,1,2), false, false, DimensionRoundingType::CEIL), // CEIL with asymmetric Padding
                                                      })),
               framework::dataset::make("Expected", { false, false, false, false, true, false, false, false, true , false, true, false, false, false})),
               input_info, output_info, pool_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLPooling3dLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), pool_info)) == expected, framework::LogLevel::ERRORS);
}


template <typename T>
using CLPooling3dLayerFixture = Pooling3dLayerValidationFixture<CLTensor, CLAccessor, CLPooling3dLayer, T>;

template <typename T>
using CLSpecialPooling3dLayerFixture = SpecialPooling3dLayerValidationFixture<CLTensor, CLAccessor, CLPooling3dLayer, T>;

template <typename T>
using CLPooling3dLayerGlobalFixture = Pooling3dLayerGlobalValidationFixture<CLTensor, CLAccessor, CLPooling3dLayer, T>;

// clang-format on
// *INDENT-ON*
TEST_SUITE(Float)
TEST_SUITE(FP32)

FIXTURE_DATA_TEST_CASE(RunSpecial, CLSpecialPooling3dLayerFixture<float>, framework::DatasetMode::ALL, datasets::Pooling3dLayerDatasetSpecial() * framework::dataset::make("DataType", DataType::F32))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLPooling3dLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small5dShapes(), combine(Pooling3dLayerDatasetFPSmall,
                                                                                                            framework::dataset::make("DataType", DataType::F32))))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLPooling3dLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::Large5dShapes(), combine(Pooling3dLayerDatasetFP,
                                                                                                            framework::dataset::make("DataType",
                                                                                                                    DataType::F32))))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

TEST_SUITE(GlobalPooling)
// *INDENT-OFF*
// clang-format off
FIXTURE_DATA_TEST_CASE(RunSmall, CLPooling3dLayerFixture<float>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(
                                    framework::dataset::make("InputShape", { TensorShape(3U, 27U, 13U, 4U),
                                                                             TensorShape(4U, 27U, 13U, 4U, 2U)
                                                                           }),
                                    framework::dataset::make("PoolingType", { PoolingType::AVG, PoolingType::L2, PoolingType::MAX })),
                                    framework::dataset::make("PoolingSize", { Size3D(27, 13, 4) })),
                                    framework::dataset::make("Strides",  Size3D(1, 1, 1))),
                                    framework::dataset::make("Paddings", Padding3D(0, 0, 0))),
                                    framework::dataset::make("ExcludePadding", false)),
                                    framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunSmallGlobal, CLPooling3dLayerGlobalFixture<float>, framework::DatasetMode::ALL,
                       combine(combine(
                                    framework::dataset::make("InputShape", { TensorShape(27U, 13U, 4U, 3U),
                                                                             TensorShape(27U, 13U, 4U, 4U, 2U)
                                                                           }),
                                    framework::dataset::make("PoolingType", { PoolingType::AVG, PoolingType::L2, PoolingType::MAX })),
                                    framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLPooling3dLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(combine(combine(
                                    framework::dataset::make("InputShape", { TensorShape(4U, 79U, 37U, 11U),
                                                                             TensorShape(4U, 79U, 37U, 11U, 2U)
                                                                           }),
                                    framework::dataset::make("PoolingType", { PoolingType::AVG, PoolingType::L2, PoolingType::MAX })),
                                    framework::dataset::make("PoolingSize", { Size3D(79, 37, 11) })),
                                    framework::dataset::make("Strides",  Size3D(1, 1, 1))),
                                    framework::dataset::make("Paddings", Padding3D(0, 0, 0))),
                                    framework::dataset::make("ExcludePadding", false)),
                                    framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
// clang-format on
// *INDENT-ON*
TEST_SUITE_END() // GlobalPooling
TEST_SUITE_END() // FP32

TEST_SUITE(FP16)

FIXTURE_DATA_TEST_CASE(RunSmall, CLPooling3dLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small5x5Shapes(), combine(Pooling3dLayerDatasetFPSmall,
                                                                                                           framework::dataset::make("DataType", DataType::F16))))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLPooling3dLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(datasets::Large5dShapes(), combine(Pooling3dLayerDatasetFP,
                                                                                                           framework::dataset::make("DataType",
                                                                                                                   DataType::F16))))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}

TEST_SUITE(GlobalPooling)
// *INDENT-OFF*
// clang-format off
FIXTURE_DATA_TEST_CASE(RunSmall, CLPooling3dLayerFixture<half>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(
                                    framework::dataset::make("InputShape", { TensorShape(3U, 27U, 13U, 4U),
                                                                             TensorShape(4U, 27U, 13U, 4U, 2U)
                                                                           }),
                                    framework::dataset::make("PoolingType", { PoolingType::AVG, PoolingType::L2, PoolingType::MAX })),
                                    framework::dataset::make("PoolingSize", { Size3D(27, 13, 4) })),
                                    framework::dataset::make("Strides",  Size3D(1, 1, 1))),
                                    framework::dataset::make("Paddings", Padding3D(0, 0, 0))),
                                    framework::dataset::make("ExcludePadding", false)),
                                    framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunSmallGlobal, CLPooling3dLayerGlobalFixture<half>, framework::DatasetMode::ALL,
                       combine(combine(
                                    framework::dataset::make("InputShape", { TensorShape(27U, 13U, 4U, 3U),
                                                                             TensorShape(27U, 13U, 4U, 4U, 2U)
                                                                           }),
                                    framework::dataset::make("PoolingType", { PoolingType::AVG, PoolingType::L2, PoolingType::MAX })),
                                    framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLPooling3dLayerFixture<half>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(combine(combine(
                                    framework::dataset::make("InputShape", { TensorShape(4U, 79U, 37U, 11U),
                                                                             TensorShape(4U, 79U, 37U, 11U, 2U)
                                                                           }),
                                    framework::dataset::make("PoolingType", { PoolingType::AVG, PoolingType::L2, PoolingType::MAX })),
                                    framework::dataset::make("PoolingSize", { Size3D(79, 37, 11) })),
                                    framework::dataset::make("Strides",  Size3D(1, 1, 1))),
                                    framework::dataset::make("Paddings", Padding3D(0, 0, 0))),
                                    framework::dataset::make("ExcludePadding", false)),
                                    framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
// clang-format on
// *INDENT-ON*
TEST_SUITE_END() // GlobalPooling
TEST_SUITE_END() // FP16
TEST_SUITE_END() // Float
TEST_SUITE_END() // Pooling3dLayer
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
