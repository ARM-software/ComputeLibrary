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
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEPooling3dLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
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

const auto Pooling3dLayerDatasetQASYMM8Small = combine(combine(combine(combine(framework::dataset::make("PoolingType", { PoolingType::MAX, PoolingType::AVG }),
                                                                               framework::dataset::make("PoolingSize", { Size3D(3, 3, 3) })),
                                                                       framework::dataset::make("Stride", { Size3D(1, 1, 1), Size3D(2, 1, 1), Size3D(1, 2, 1), Size3D(2, 2, 1) })),
                                                               framework::dataset::make("Padding", { Padding3D(0, 0, 0), Padding3D(1, 1, 1), Padding3D(1, 0, 0) })),
                                                       framework::dataset::make("ExcludePadding", { true }));

const auto Pooling3dLayerDatasetQASYMM8Large = combine(combine(combine(combine(framework::dataset::make("PoolingType", { PoolingType::MAX, PoolingType::AVG }),
                                                                               framework::dataset::make("PoolingSize", { Size3D(3, 3, 3) })),
                                                                       framework::dataset::make("Stride", { Size3D(1, 1, 1), Size3D(2, 2, 1) })),
                                                               framework::dataset::make("Padding", { Padding3D(0, 0, 0), Padding3D(1, 1, 0) })),
                                                       framework::dataset::make("ExcludePadding", { true }));

using ShapeDataset = framework::dataset::ContainerDataset<std::vector<TensorShape>>;

constexpr AbsoluteTolerance<float> tolerance_f32(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for 32-bit floating-point type */
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
constexpr AbsoluteTolerance<float> tolerance_f16(0.01f);     /**< Tolerance value for comparing reference's output against implementation's output for 16-bit floating-point type */
#endif                                                       /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(1);   /**< Tolerance value for comparing reference's output against implementation's output for unsigned 8-bit asymmetric type */
constexpr AbsoluteTolerance<int8_t>  tolerance_qasymm8_s(1); /**< Tolerance value for comparing reference's output against implementation's output for signed 8-bit asymmetric type */

const auto qasymm8_in_qinfo_dataset  = framework::dataset::make("InputQuantInfo", { QuantizationInfo(.2f, 10) });
const auto qasymm8_out_qinfo_dataset = framework::dataset::make("OutputQuantInfo",
{
    QuantizationInfo(.2f, 10), // Same qinfo
    QuantizationInfo(.1f, 5),  // Multiplier <= 1
    QuantizationInfo(2.f, 3)   // Multiplier > 1
});

const auto qasymm8_signed_in_qinfo_dataset  = framework::dataset::make("InputQuantInfo", { QuantizationInfo(.2f, -10) });
const auto qasymm8_signed_out_qinfo_dataset = framework::dataset::make("OutputQuantInfo",
{
    QuantizationInfo(.2f, -10), // Same qinfo
    QuantizationInfo(.1f, -5),  // Multiplier <= 1
    QuantizationInfo(2.f, -3)   // Multiplier > 1
});

} //namespace

TEST_SUITE(NEON)
TEST_SUITE(Pooling3dLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
    framework::dataset::make("InputInfo", { TensorInfo(TensorShape(2U, 27U, 13U, 4U, 3U), 1, DataType::F32, DataLayout::NDHWC),     // Mismatching data type
                                            TensorInfo(TensorShape(2U, 27U, 13U, 4U, 2U), 1, DataType::F32, DataLayout::NDHWC),     // Invalid pad/size combination
                                            TensorInfo(TensorShape(2U, 27U, 13U, 4U, 2U), 1, DataType::F32, DataLayout::NDHWC),     // Invalid pad/size combination
                                            TensorInfo(TensorShape(2U, 27U, 13U, 4U, 3U), 1, DataType::F32, DataLayout::NDHWC),     // Invalid output shape
                                            TensorInfo(TensorShape(5U, 13U, 15U, 2U, 3U), 1, DataType::F32, DataLayout::NDHWC),     // Global Pooling
                                            TensorInfo(TensorShape(13U,13U, 5U, 1U, 2U),  1, DataType::F32, DataLayout::NDHWC),     // Invalid output Global Pooling
                                            TensorInfo(TensorShape(5U, 13U, 13U, 4U, 4U), 1, DataType::F32, DataLayout::NDHWC),
                                            TensorInfo(TensorShape(5U, 13U, 13U, 4U, 4U), 1, DataType::F32, DataLayout::NDHWC),     // Invalid data type
                                            TensorInfo(TensorShape(5U, 13U, 13U, 4U, 4U), 1, DataType::F32, DataLayout::NHWC),      // Invalid data layout
                                            TensorInfo(TensorShape(5U, 13U, 13U, 5U, 4U), 1, DataType::F32, DataLayout::NDHWC),
                                            TensorInfo(TensorShape(1U, 16U,  1U, 3U, 4U), 1, DataType::F32, DataLayout::NDHWC),
                                            TensorInfo(TensorShape(5U, 13U, 13U, 4U, 3U), 1, DataType::F32, DataLayout::NDHWC),
                                            TensorInfo(TensorShape(5U, 13U, 13U, 4U, 2U), 1, DataType::F32, DataLayout::NDHWC),
                                            TensorInfo(TensorShape(5U, 13U, 13U, 4U, 3U), 1, DataType::F32, DataLayout::NDHWC),
                                          }),
    framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(2U, 25U, 11U, 3U, 3U), 1, DataType::F16, DataLayout::NDHWC),
                                            TensorInfo(TensorShape(2U, 30U, 11U, 3U, 2U), 1, DataType::F32, DataLayout::NDHWC),
                                            TensorInfo(TensorShape(2U, 25U, 16U, 3U, 2U), 1, DataType::F32, DataLayout::NDHWC),
                                            TensorInfo(TensorShape(2U, 27U, 13U, 3U, 3U), 1, DataType::F32, DataLayout::NDHWC),
                                            TensorInfo(TensorShape(5U,  1U,  1U, 1U, 3U), 1, DataType::F32, DataLayout::NDHWC),            // Global pooling applied
                                            TensorInfo(TensorShape(5U,  2U,  2U, 2U, 2U), 1, DataType::F32, DataLayout::NDHWC),            // Invalid output Global Pooling
                                            TensorInfo(TensorShape(5U, 12U, 12U, 3U, 4U), 1, DataType::F32, DataLayout::NDHWC),
                                            TensorInfo(TensorShape(5U, 12U, 12U, 3U, 4U), 1, DataType::QASYMM8, DataLayout::NDHWC),        // Invalid data type
                                            TensorInfo(TensorShape(5U, 12U, 12U, 3U, 4U), 1, DataType::F32, DataLayout::NDHWC),            // Invalid data layout
                                            TensorInfo(TensorShape(5U,  1U,  1U, 1U, 4U), 1, DataType::F32, DataLayout::NDHWC),
                                            TensorInfo(TensorShape(1U, 15U, 1U, 2U, 4U), 1, DataType::F32, DataLayout::NDHWC),             // size larger than height
                                            TensorInfo(TensorShape(5U, 6U, 6U, 2U, 3U),  1, DataType::F32, DataLayout::NDHWC),
                                            TensorInfo(TensorShape(5U, 6U, 6U, 2U, 2U),  1, DataType::F32, DataLayout::NDHWC),
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
                                            Pooling3dLayerInfo(PoolingType::AVG, 2, Size3D(1U, 1U, 1U), Padding3D(), false),
                                            Pooling3dLayerInfo(PoolingType::AVG),
                                            Pooling3dLayerInfo(PoolingType::MAX, 2, Size3D(1, 1, 2), Padding3D(0, 0, 0), false),
                                            Pooling3dLayerInfo(PoolingType::AVG, 2, Size3D(2U, 2U, 2U), Padding3D(), false),
                                            Pooling3dLayerInfo(PoolingType::AVG, 1, Size3D(2U, 2U, 2U), Padding3D(2, 2, 2), true),  // pool size is equal to the padding size
                                            Pooling3dLayerInfo(PoolingType::AVG, 1, Size3D(2U, 2U, 2U), Padding3D(2, 2, 2), false), // pool size is equal to the padding size
                                            Pooling3dLayerInfo(PoolingType::AVG, 3, Size3D(2U, 2U, 2U), Padding3D(2,1,2,2,1,2), false, false, DimensionRoundingType::CEIL), // CEIL with asymmetric Padding
                                            })),
    framework::dataset::make("Expected", { false, false, false, false, true, false, false, false, false, true , false, true, false, false, false})),
    input_info, output_info, pool_info, expected)
{
    bool is_valid = bool(NEPooling3dLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), pool_info));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEPoolingLayer3dFixture = Pooling3dLayerValidationFixture<Tensor, Accessor, NEPooling3dLayer, T>;

template <typename T>
using NESpecial3dPoolingLayerFixture = SpecialPooling3dLayerValidationFixture<Tensor, Accessor, NEPooling3dLayer, T>;

template <typename T>
using NEPooling3dLayerGlobalFixture = Pooling3dLayerGlobalValidationFixture<Tensor, Accessor, NEPooling3dLayer, T>;

// clang-format on
// *INDENT-ON*
TEST_SUITE(Float)
TEST_SUITE(FP32)

FIXTURE_DATA_TEST_CASE(RunSpecial, NESpecial3dPoolingLayerFixture<float>, framework::DatasetMode::ALL, datasets::Pooling3dLayerDatasetSpecial() * framework::dataset::make("DataType", DataType::F32))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunSmall, NEPoolingLayer3dFixture<float>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small5dShapes(), combine(Pooling3dLayerDatasetFPSmall,
                                                                                                            framework::dataset::make("DataType", DataType::F32))))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEPoolingLayer3dFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(datasets::Large5dShapes(), combine(Pooling3dLayerDatasetFPSmall, framework::dataset::make("DataType", DataType::F32))))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}

TEST_SUITE(GlobalPooling)
// *INDENT-OFF*
// clang-format off
FIXTURE_DATA_TEST_CASE(RunSmall, NEPoolingLayer3dFixture<float>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(
                                    framework::dataset::make("InputShape", { TensorShape(3U, 27U, 13U, 4U),
                                                                             TensorShape(4U, 27U, 13U, 4U, 2U)
                                                                           }),
                                    framework::dataset::make("PoolingType", { PoolingType::AVG, PoolingType::L2, PoolingType::MAX })),
                                    framework::dataset::make("PoolingSize", { Size3D(27, 13, 4) })),
                                    framework::dataset::make("Strides",  Size3D(1, 1, 1))),
                                    framework::dataset::make("Paddings", Padding3D(0, 0, 0))),
                                    framework::dataset::make("ExcludePadding", {false, true})),
                                    framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunGlobalSmall, NEPooling3dLayerGlobalFixture<float>, framework::DatasetMode::ALL,
                       combine(combine(
                                    framework::dataset::make("InputShape", { TensorShape(27U, 13U, 4U, 3U),
                                                                             TensorShape(27U, 13U, 4U, 4U, 2U)
                                                                           }),
                                    framework::dataset::make("PoolingType", { PoolingType::AVG, PoolingType::L2, PoolingType::MAX })),
                                    framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEPoolingLayer3dFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(combine(combine(
                                    framework::dataset::make("InputShape", { TensorShape(4U, 79U, 37U, 11U),
                                                                             TensorShape(4U, 79U, 37U, 11U, 2U)
                                                                           }),
                                    framework::dataset::make("PoolingType", { PoolingType::AVG, PoolingType::L2, PoolingType::MAX })),
                                    framework::dataset::make("PoolingSize", { Size3D(79, 37, 11) })),
                                    framework::dataset::make("Strides",  Size3D(1, 1, 1))),
                                    framework::dataset::make("Paddings", Padding3D(0, 0, 0))),
                                    framework::dataset::make("ExcludePadding", {false, true})),
                                    framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}

TEST_SUITE_END() // GlobalPooling
TEST_SUITE_END() // FP32

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)

FIXTURE_DATA_TEST_CASE(RunSmall, NEPoolingLayer3dFixture<half>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small5x5Shapes(), combine(Pooling3dLayerDatasetFPSmall,
                                                                                                           framework::dataset::make("DataType", DataType::F16))))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}


FIXTURE_DATA_TEST_CASE(RunLarge, NEPoolingLayer3dFixture<half>, framework::DatasetMode::NIGHTLY, combine(datasets::Large5dShapes(), combine(Pooling3dLayerDatasetFP,
                                                                                                           framework::dataset::make("DataType",
                                                                                                                   DataType::F16))))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}

TEST_SUITE(GlobalPooling)
// *INDENT-OFF*
// clang-format off
FIXTURE_DATA_TEST_CASE(RunSmall, NEPoolingLayer3dFixture<half>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(
                                    framework::dataset::make("InputShape", { TensorShape(3U, 27U, 13U, 4U),
                                                                             TensorShape(4U, 27U, 13U, 4U, 2U)
                                                                           }),
                                    framework::dataset::make("PoolingType", { PoolingType::AVG, PoolingType::L2, PoolingType::MAX })),
                                    framework::dataset::make("PoolingSize", { Size3D(27, 13, 4) })),
                                    framework::dataset::make("Strides",  Size3D(1, 1, 1))),
                                    framework::dataset::make("Paddings", Padding3D(0, 0, 0))),
                                    framework::dataset::make("ExcludePadding", {false, true})),
                                    framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}


FIXTURE_DATA_TEST_CASE(RunSmallGlobal, NEPooling3dLayerGlobalFixture<half>, framework::DatasetMode::ALL,
                       combine(combine(
                                    framework::dataset::make("InputShape", { TensorShape(27U, 13U, 4U, 3U),
                                                                             TensorShape(27U, 13U, 4U, 4U, 2U)
                                                                           }),
                                    framework::dataset::make("PoolingType", { PoolingType::AVG, PoolingType::L2, PoolingType::MAX })),
                                    framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEPoolingLayer3dFixture<half>, framework::DatasetMode::NIGHTLY,
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
    validate(Accessor(_target), _reference, tolerance_f16);
}

// clang-format on
// *INDENT-ON*
TEST_SUITE_END() // GlobalPooling
TEST_SUITE_END() // FP16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
TEST_SUITE_END() // Float
TEST_SUITE(Quantized)

template <typename T>
using NEPooling3dLayerQuantizedFixture = Pooling3dLayerValidationQuantizedFixture<Tensor, Accessor, NEPooling3dLayer, T>;

TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEPooling3dLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::Small5dShapes(),
                                                                                                                       combine(Pooling3dLayerDatasetQASYMM8Small,
                                                                                                                               framework::dataset::make("DataType", DataType::QASYMM8))),
                                                                                                                       qasymm8_in_qinfo_dataset),
                                                                                                                       qasymm8_out_qinfo_dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEPooling3dLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::Large5dShapes(),
                                                                                                                       combine(Pooling3dLayerDatasetQASYMM8Large,
                                                                                                                               framework::dataset::make("DataType", DataType::QASYMM8))),
                                                                                                                       qasymm8_in_qinfo_dataset),
                                                                                                                       qasymm8_out_qinfo_dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}

TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)

FIXTURE_DATA_TEST_CASE(RunSmall, NEPooling3dLayerQuantizedFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::Small5dShapes(),
                                                                                                                      combine(Pooling3dLayerDatasetQASYMM8Small,
                                                                                                                              framework::dataset::make("DataType", DataType::QASYMM8_SIGNED))),
                                                                                                                      qasymm8_signed_in_qinfo_dataset),
                                                                                                                      qasymm8_signed_out_qinfo_dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8_s);
}

TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // Quantized
TEST_SUITE_END() // Pooling3dLayer
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
