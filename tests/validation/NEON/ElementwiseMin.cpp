/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEElementwiseOperations.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ElementwiseOperationsFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr RelativeTolerance<float>  tolerance_fp32(0.000001f);
constexpr AbsoluteTolerance<int8_t> tolerance_qasymm8_signed(1);
/** Input data sets **/
const auto ElementwiseMinQASYMM8Dataset = combine(combine(framework::dataset::make("DataType", DataType::QASYMM8), framework::dataset::make("DataType", DataType::QASYMM8)),
                                                  framework::dataset::make("DataType",
                                                                           DataType::QASYMM8));

const auto ElementwiseMaxQASYMM8SignedDataset = combine(combine(framework::dataset::make("DataType", DataType::QASYMM8_SIGNED), framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                                                        framework::dataset::make("DataType",
                                                                                 DataType::QASYMM8_SIGNED));

const auto ElementwiseMinS32Dataset = combine(combine(framework::dataset::make("DataType", DataType::S32), framework::dataset::make("DataType", DataType::S32)), framework::dataset::make("DataType",
                                              DataType::S32));
const auto ElementwiseMinS16Dataset = combine(combine(framework::dataset::make("DataType", { DataType::S16 }), framework::dataset::make("DataType", DataType::S16)),
                                              framework::dataset::make("DataType", DataType::S16));
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
const auto ElementwiseMinFP16Dataset = combine(combine(framework::dataset::make("DataType", DataType::F16), framework::dataset::make("DataType", DataType::F16)),
                                               framework::dataset::make("DataType", DataType::F16));
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
const auto ElementwiseMinFP32Dataset = combine(combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::F32)),
                                               framework::dataset::make("DataType", DataType::F32));
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(ElementwiseMin)

template <typename T>
using NEElementwiseMinFixture = ElementwiseMinValidationFixture<Tensor, Accessor, NEElementwiseMin, T>;

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
               framework::dataset::make("Input1Info", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S32),
                                                        TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::S32),
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S32),                // Invalid data type combination
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),                // Mismatching shapes
                                                        TensorInfo(TensorShape(4U, 4U, 2U), 1, DataType::QASYMM8_SIGNED),       // Ok
                                                        TensorInfo(TensorShape(4U, 4U, 2U), 1, DataType::QASYMM8_SIGNED),       // Mismatching types, cannot mix QASYMM8_SIGNED with QASYMM8
                                                      }),
               framework::dataset::make("Input2Info",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S32),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::S32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(4U, 4U, 2U), 1, DataType::QASYMM8_SIGNED),
                                                       TensorInfo(TensorShape(4U, 4U, 2U), 1, DataType::QASYMM8),
                                                     })),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S32),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::S32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S32),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(4U, 4U, 2U), 1, DataType::QASYMM8_SIGNED),
                                                       TensorInfo(TensorShape(4U, 4U, 2U), 1, DataType::QASYMM8_SIGNED),
                                                     })),
               framework::dataset::make("Expected", { true, true, true, false,
                                                      false,true,false})),
               input1_info, input2_info, output_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(NEElementwiseMin::validate(
                                    &input1_info.clone()->set_is_resizable(false),
                                    &input2_info.clone()->set_is_resizable(false),
                                    &output_info.clone()->set_is_resizable(false))) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE(S32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEElementwiseMinFixture<int32_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallShapes(), ElementwiseMinS32Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S32

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEElementwiseMinFixture<int16_t>, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), ElementwiseMinS16Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S16

template <typename T>
using NEElementwiseMinQuantizedFixture = ElementwiseMinValidationQuantizedFixture<Tensor, Accessor, NEElementwiseMin, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
template <typename T>
using NEElementwiseMinQuantizedBroadcastFixture = ElementwiseMinQuantizedBroadcastValidationFixture<Tensor, Accessor, NEElementwiseMin, T>;

FIXTURE_DATA_TEST_CASE(RunSmallBroadcast, NEElementwiseMinQuantizedBroadcastFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapesBroadcast(),
                       ElementwiseMinQASYMM8Dataset),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(5.f / 255.f, 20) })),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(2.f / 255.f, 10) })),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(1.f / 255.f, 5) })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunSmall, NEElementwiseMinQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                       ElementwiseMinQASYMM8Dataset),
                                                                                                                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(5.f / 255.f, 20) })),
                                                                                                                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(2.f / 255.f, 10) })),
                                                                                                                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(1.f / 255.f, 5) }))

                      )
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32, 0.01);
}
TEST_SUITE_END()

TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall, NEElementwiseMinQuantizedFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                      ElementwiseMaxQASYMM8SignedDataset),
                                                                                                                      framework::dataset::make("QuantizationInfo", { QuantizationInfo(10.f, 20) })),
                                                                                                                      framework::dataset::make("QuantizationInfo", { QuantizationInfo(1.f, 0) })),
                                                                                                                      framework::dataset::make("QuantizationInfo", { QuantizationInfo(2.f, -27) })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qasymm8_signed);
}

TEST_SUITE_END()

TEST_SUITE_END()

TEST_SUITE(Float)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(F16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEElementwiseMinFixture<half>, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), ElementwiseMinFP16Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // F16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEElementwiseMinFixture<float>, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), ElementwiseMinFP32Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

template <typename T>
using NEElementwiseMinBroadcastFixture = ElementwiseMinBroadcastValidationFixture<Tensor, Accessor, NEElementwiseMin, T>;

FIXTURE_DATA_TEST_CASE(RunSmallBroadcast, NEElementwiseMinBroadcastFixture<float>, framework::DatasetMode::ALL, combine(datasets::SmallShapesBroadcast(),
                                                                                                                        ElementwiseMinFP32Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // F32
TEST_SUITE_END() // Float

TEST_SUITE_END() // ElementwiseMin
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
