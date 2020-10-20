/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLElementwiseOperations.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ConvertPolicyDataset.h"
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
RelativeTolerance<float> tolerance_fp32(0.000001f);
RelativeTolerance<float> tolerance_fp16(0.001f);

/** Input data sets **/
const auto ElementwiseMinU8Dataset = combine(combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::U8)), framework::dataset::make("DataType",
                                             DataType::U8));
const auto ElementwiseMinQASYMM8Dataset = combine(combine(framework::dataset::make("DataType", DataType::QASYMM8), framework::dataset::make("DataType", DataType::QASYMM8)),
                                                  framework::dataset::make("DataType",
                                                                           DataType::QASYMM8));
const auto ElementwiseMinQASYMM8SignedDataset = combine(combine(framework::dataset::make("DataType", DataType::QASYMM8_SIGNED), framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                                                        framework::dataset::make("DataType",
                                                                                 DataType::QASYMM8_SIGNED));
const auto ElementwiseMinQSYMM16Dataset = combine(combine(framework::dataset::make("DataType", DataType::QSYMM16), framework::dataset::make("DataType", DataType::QSYMM16)),
                                                  framework::dataset::make("DataType",
                                                                           DataType::QSYMM16));
const auto ElementwiseMinS16Dataset = combine(combine(framework::dataset::make("DataType", { DataType::U8, DataType::S16 }), framework::dataset::make("DataType", DataType::S16)),
                                              framework::dataset::make("DataType", DataType::S16));
const auto ElementwiseMinFP16Dataset = combine(combine(framework::dataset::make("DataType", DataType::F16), framework::dataset::make("DataType", DataType::F16)),
                                               framework::dataset::make("DataType", DataType::F16));
const auto ElementwiseMinFP32Dataset = combine(combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::F32)),
                                               framework::dataset::make("DataType", DataType::F32));
const auto EmptyActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{ ActivationLayerInfo() });
const auto ActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 0.75f, 0.25f),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC, 0.75f, 0.25f)
});
} // namespace

TEST_SUITE(CL)
TEST_SUITE(ElementwiseMin)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
               framework::dataset::make("Input1Info", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),      // Invalid data type combination
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),     // Mismatching shapes
                                                      }),
               framework::dataset::make("Input2Info",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("Expected", { true, true, false, false})),
               input1_info, input2_info, output_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLElementwiseMin::validate(&input1_info.clone()->set_is_resizable(false), &input2_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false))) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLElementwiseMinFixture = ElementwiseMinValidationFixture<CLTensor, CLAccessor, CLElementwiseMin, T>;

TEST_SUITE(Integer)
TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLElementwiseMinFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallShapes(), ElementwiseMinU8Dataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLElementwiseMinFixture<int16_t>, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), ElementwiseMinS16Dataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using CLElementwiseMinQuantizedFixture = ElementwiseMinValidationQuantizedFixture<CLTensor, CLAccessor, CLElementwiseMin, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLElementwiseMinQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                       ElementwiseMinQASYMM8Dataset),
                                                                                                                       framework::dataset::make("Src0QInfo", { QuantizationInfo(5.f / 255.f, 20) })),
                                                                                                                       framework::dataset::make("Src1QInfo", { QuantizationInfo(2.f / 255.f, 10) })),
                                                                                                                       framework::dataset::make("OutQInfo", { QuantizationInfo(1.f / 255.f, 5) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp32, 0.01);
}
TEST_SUITE_END()
TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall, CLElementwiseMinQuantizedFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                      ElementwiseMinQASYMM8SignedDataset),
                                                                                                                      framework::dataset::make("Src0QInfo", { QuantizationInfo(5.f / 255.f, 20) })),
                                                                                                                      framework::dataset::make("Src1QInfo", { QuantizationInfo(2.f / 255.f, 10) })),
                                                                                                                      framework::dataset::make("OutQInfo", { QuantizationInfo(1.f / 255.f, 5) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()
TEST_SUITE(QSYMM16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLElementwiseMinQuantizedFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                       ElementwiseMinQSYMM16Dataset),
                                                                                                                       framework::dataset::make("SrcQInfo0", { QuantizationInfo(1.f / 32768.f, 0), QuantizationInfo(5.f / 32768.f, 0) })),
                                                                                                                       framework::dataset::make("SrcQInfo1", { QuantizationInfo(2.f / 32768.f, 0), QuantizationInfo(5.f / 32768.f, 0) })),
                                                                                                                       framework::dataset::make("OutQInfo", { QuantizationInfo(5.f / 32768.f, 0) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using CLElementwiseMinFloatFixture = ElementwiseMinValidationFloatFixture<CLTensor, CLAccessor, CLElementwiseMin, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLElementwiseMinFloatFixture<half>, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), ElementwiseMinFP16Dataset), EmptyActivationFunctionsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp16, 0.01);
}
FIXTURE_DATA_TEST_CASE(RunWithActivation, CLElementwiseMinFloatFixture<half>, framework::DatasetMode::ALL, combine(combine(datasets::TinyShapes(), ElementwiseMinFP16Dataset),
                                                                                                                   ActivationFunctionsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp16, 0.01);
}
TEST_SUITE_END()

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLElementwiseMinFloatFixture<float>, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), ElementwiseMinFP32Dataset),
                                                                                                           EmptyActivationFunctionsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp32);
}
FIXTURE_DATA_TEST_CASE(RunWithActivation, CLElementwiseMinFloatFixture<float>, framework::DatasetMode::ALL, combine(combine(datasets::TinyShapes(), ElementwiseMinFP32Dataset),
                                                                                                                    ActivationFunctionsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp32);
}
template <typename T>
using CLElementwiseMinBroadcastFloatFixture = ElementwiseMinBroadcastValidationFloatFixture<CLTensor, CLAccessor, CLElementwiseMin, T>;

FIXTURE_DATA_TEST_CASE(RunSmallBroadcast, CLElementwiseMinBroadcastFloatFixture<float>, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapesBroadcast(),
                       ElementwiseMinFP32Dataset),
                       EmptyActivationFunctionsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp32);
}
FIXTURE_DATA_TEST_CASE(RunWithActivationBroadcast, CLElementwiseMinBroadcastFloatFixture<float>, framework::DatasetMode::ALL, combine(combine(datasets::TinyShapesBroadcast(),
                       ElementwiseMinFP32Dataset),
                       ActivationFunctionsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp32);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END() // ElementwiseMin
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
