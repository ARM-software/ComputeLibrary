/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLPoolingLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/PoolingTypesDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/PoolingLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Failing data set */
const auto PoolingLayerDatasetSpecial = ((((framework::dataset::make("Shape", TensorShape{ 60U, 52U, 3U, 5U })
                                            * framework::dataset::make("PoolType", PoolingType::AVG))
                                           * framework::dataset::make("PoolingSize", 100))
                                          * framework::dataset::make("PadStride", PadStrideInfo(5, 5, 50, 50)))
                                         * framework::dataset::make("ExcludePadding", true));
/** Input data set for floating-point data types */
const auto PoolingLayerDatasetFP = combine(combine(combine(datasets::PoolingTypes(), framework::dataset::make("PoolingSize", { 2, 3, 4, 7, 9 })),
                                                   framework::dataset::make("PadStride", { PadStrideInfo(1, 1, 0, 0), PadStrideInfo(2, 1, 0, 0), PadStrideInfo(1, 2, 1, 1), PadStrideInfo(2, 2, 1, 0) })),
                                           framework::dataset::make("ExcludePadding", { true, false }));

/** Input data set for fixed-point data types */
const auto PoolingLayerDatasetQS = combine(combine(combine(framework::dataset::make("PoolingType", { PoolingType::MAX, PoolingType::AVG }), framework::dataset::make("PoolingSize", { 2, 3 })),
                                                   framework::dataset::make("PadStride", { PadStrideInfo(1, 1, 0, 0), PadStrideInfo(2, 1, 0, 0), PadStrideInfo(1, 2, 1, 1), PadStrideInfo(2, 2, 1, 0) })),
                                           framework::dataset::make("ExcludePadding", { true, false }));

/** Input data set for asymmetric data type */
const auto PoolingLayerDatasetQASYMM8 = combine(combine(combine(framework::dataset::make("PoolingType", { PoolingType::MAX, PoolingType::AVG }), framework::dataset::make("PoolingSize", { 2, 3 })),
                                                        framework::dataset::make("PadStride", { PadStrideInfo(1, 1, 0, 0), PadStrideInfo(2, 1, 0, 0), PadStrideInfo(1, 2, 1, 1), PadStrideInfo(2, 2, 1, 0) })),
                                                framework::dataset::make("ExcludePadding", { true, false }));

constexpr AbsoluteTolerance<float>   tolerance_f32(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for 32-bit floating-point type */
constexpr AbsoluteTolerance<float>   tolerance_f16(0.01f);  /**< Tolerance value for comparing reference's output against implementation's output for 16-bit floating-point type */
constexpr AbsoluteTolerance<float>   tolerance_qs16(6);     /**< Tolerance value for comparing reference's output against implementation's output for 16-bit fixed-point type */
constexpr AbsoluteTolerance<float>   tolerance_qs8(3);      /**< Tolerance value for comparing reference's output against implementation's output for 8-bit fixed-point type */
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(1);  /**< Tolerance value for comparing reference's output against implementation's output for 8-bit asymmetric type */
} // namespace

TEST_SUITE(CL)
TEST_SUITE(PoolingLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, 0),     // Mismatching data type
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, 0),     // Window shrink
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::QS8, 4),     // Mismatching fixed point position
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::QS16, 11),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, 0),     // Invalid pad/size combination
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, 0),     // Invalid pad/size combination
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::QASYMM8, 0), // Invalid parameters
                                                       TensorInfo(TensorShape(15U, 13U, 5U), 1, DataType::F32, 0),     // Non-rectangular Global Pooling
                                                       TensorInfo(TensorShape(13U, 13U, 5U), 1, DataType::F32, 0),     // Invalid output Global Pooling
                                                       TensorInfo(TensorShape(13U, 13U, 5U), 1, DataType::F32, 0),
                                                     }),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F16, 0),
                                                       TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32, 0),
                                                       TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::QS8, 5),
                                                       TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::QS16, 11),
                                                       TensorInfo(TensorShape(30U, 11U, 2U), 1, DataType::F32, 0),
                                                       TensorInfo(TensorShape(25U, 16U, 2U), 1, DataType::F32, 0),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::QASYMM8, 0),
                                                       TensorInfo(TensorShape(1U, 1U, 5U), 1, DataType::F32, 0),
                                                       TensorInfo(TensorShape(2U, 2U, 5U), 1, DataType::F32, 0),
                                                       TensorInfo(TensorShape(1U, 1U, 5U), 1, DataType::F32, 0),
                                                     })),
               framework::dataset::make("PoolInfo",  { PoolingLayerInfo(PoolingType::AVG, 3, PadStrideInfo(1, 1, 0, 0)),
                                                       PoolingLayerInfo(PoolingType::AVG, 3, PadStrideInfo(1, 1, 0, 0)),
                                                       PoolingLayerInfo(PoolingType::AVG, 3, PadStrideInfo(1, 1, 0, 0)),
                                                       PoolingLayerInfo(PoolingType::AVG, 3, PadStrideInfo(1, 1, 0, 0)),
                                                       PoolingLayerInfo(PoolingType::AVG, 2, PadStrideInfo(1, 1, 2, 0)),
                                                       PoolingLayerInfo(PoolingType::AVG, 2, PadStrideInfo(1, 1, 0, 2)),
                                                       PoolingLayerInfo(PoolingType::L2, 3, PadStrideInfo(1, 1, 0, 0)),
                                                       PoolingLayerInfo(PoolingType::AVG),
                                                       PoolingLayerInfo(PoolingType::MAX),
                                                       PoolingLayerInfo(PoolingType::AVG),
                                                      })),
               framework::dataset::make("Expected", { false, false, false, true, false, false, false, false, false, true })),
               input_info, output_info, pool_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLPoolingLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), pool_info)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLPoolingLayerFixture = PoolingLayerValidationFixture<CLTensor, CLAccessor, CLPoolingLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSpecial, CLPoolingLayerFixture<float>, framework::DatasetMode::ALL, PoolingLayerDatasetSpecial * framework::dataset::make("DataType", DataType::F32))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunSmall, CLPoolingLayerFixture<float>, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), combine(PoolingLayerDatasetFP, framework::dataset::make("DataType",
                                                                                                    DataType::F32))))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLPoolingLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), combine(PoolingLayerDatasetFP, framework::dataset::make("DataType",
                                                                                                        DataType::F32))))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END()

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLPoolingLayerFixture<half>, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), combine(PoolingLayerDatasetFP,
                                                                                                   framework::dataset::make("DataType", DataType::F16))))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLPoolingLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), combine(PoolingLayerDatasetFP,
                                                                                                       framework::dataset::make("DataType", DataType::F16))))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using CLPoolingLayerFixedPointFixture = PoolingLayerValidationFixedPointFixture<CLTensor, CLAccessor, CLPoolingLayer, T>;

TEST_SUITE(FixedPoint)
TEST_SUITE(QS8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLPoolingLayerFixedPointFixture<int8_t>, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), combine(PoolingLayerDatasetQS,
                                                                                                                       framework::dataset::make("DataType", DataType::QS8))),
                                                                                                               framework::dataset::make("FractionalBits", 1, 4)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qs8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLPoolingLayerFixedPointFixture<int8_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), combine(PoolingLayerDatasetQS,
                                                                                                                   framework::dataset::make("DataType", DataType::QS8))),
                                                                                                                   framework::dataset::make("FractionalBits", 1, 4)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qs8);
}
TEST_SUITE_END()

TEST_SUITE(QS16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLPoolingLayerFixedPointFixture<int16_t>, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), combine(PoolingLayerDatasetQS,
                                                                                                                        framework::dataset::make("DataType", DataType::QS16))),
                                                                                                                framework::dataset::make("FractionalBits", 1, 12)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qs16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLPoolingLayerFixedPointFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), combine(PoolingLayerDatasetQS,
                                                                                                                    framework::dataset::make("DataType", DataType::QS16))),
                                                                                                                    framework::dataset::make("FractionalBits", 1, 12)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qs16);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE(Quantized)

template <typename T>
using CLPoolingLayerQuantizedFixture = PoolingLayerValidationQuantizedFixture<CLTensor, CLAccessor, CLPoolingLayer, T>;

TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLPoolingLayerQuantizedFixture<uint8_t>, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), combine(PoolingLayerDatasetQASYMM8,
                                                                                                                       framework::dataset::make("DataType", DataType::QASYMM8))),
                                                                                                               framework::dataset::make("QuantizationInfo", { QuantizationInfo(2.f / 255, 127),
                                                                                                                       QuantizationInfo(7.f / 255, 123)
                                                                                                                                                            })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLPoolingLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), combine(PoolingLayerDatasetQASYMM8,
                                                                                                                   framework::dataset::make("DataType", DataType::QASYMM8))),
                                                                                                                   framework::dataset::make("QuantizationInfo", { QuantizationInfo(1.f / 255, 0) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
