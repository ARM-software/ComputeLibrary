/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLPReluLayer.h"
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
const auto PReluLayerU8Dataset = combine(combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::U8)),
                                         framework::dataset::make("DataType",
                                                                  DataType::U8));
const auto PReluLayerQASYMM8Dataset = combine(combine(framework::dataset::make("DataType", DataType::QASYMM8), framework::dataset::make("DataType", DataType::QASYMM8)),
                                              framework::dataset::make("DataType",
                                                                       DataType::QASYMM8));
const auto PReluLayerQASYMM8SIGNEDDataset = combine(combine(framework::dataset::make("DataType", DataType::QASYMM8_SIGNED), framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                                                    framework::dataset::make("DataType",
                                                                             DataType::QASYMM8_SIGNED));
const auto PReluLayerS16Dataset = combine(combine(framework::dataset::make("DataType", { DataType::S16 }), framework::dataset::make("DataType", DataType::S16)),
                                          framework::dataset::make("DataType", DataType::S16));
const auto PReluLayerFP16Dataset = combine(combine(framework::dataset::make("DataType", DataType::F16), framework::dataset::make("DataType", DataType::F16)),
                                           framework::dataset::make("DataType", DataType::F16));
const auto PReluLayerFP32Dataset = combine(combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::F32)),
                                           framework::dataset::make("DataType", DataType::F32));
} // namespace

TEST_SUITE(CL)
TEST_SUITE(PReluLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
               framework::dataset::make("Input1Info", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),      // Invalid data type combination
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),     // Mismatching shapes
                                                      }),
               framework::dataset::make("Input2Info",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("Expected", { true, false, false})),
               input1_info, input2_info, output_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLPReluLayer::validate(&input1_info.clone()->set_is_resizable(false), &input2_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false))) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE(InPlace)
TEST_CASE(Validate, framework::DatasetMode::ALL)
{
    // PRelu operaotr should be able to take nullptr as output and do the in-place computation.
    // Shape and data type are selected randomly since they shouldn't matter
    const auto tensor_info = TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32);
    const auto result      = arm_compute::CLPReluLayer::validate(&tensor_info, &tensor_info, nullptr);
    ARM_COMPUTE_EXPECT(bool(result) == true, framework::LogLevel::ERRORS);
}

SimpleTensor<float> compute_float_reference(const TensorInfo &tensor_info)
{
    SimpleTensor<float> ref_src1{ tensor_info.tensor_shape(), tensor_info.data_type() };
    SimpleTensor<float> ref_src2{ tensor_info.tensor_shape(), tensor_info.data_type() };
    SimpleTensor<float> ref_dst{ tensor_info.tensor_shape(), tensor_info.data_type() };

    library->fill_tensor_uniform(ref_src1, 0);
    library->fill_tensor_uniform(ref_src2, 1);

    return reference::arithmetic_operation<float>(ArithmeticOperation::PRELU, ref_src1, ref_src2, ref_dst);
}

void compute_float_target_in_place(CLTensor &src1, CLTensor &src2, bool use_nullptr_output)
{
    auto fn = arm_compute::CLPReluLayer{};
    fn.configure(&src1, &src2, use_nullptr_output ? nullptr : &src1);

    src1.allocator()->allocate();
    src2.allocator()->allocate();

    library->fill_tensor_uniform(CLAccessor(src1), 0);
    library->fill_tensor_uniform(CLAccessor(src2), 1);

    fn.run();
}

TEST_CASE(ComputeWithNullPtr, framework::DatasetMode::ALL)
{
    const auto tensor_info = TensorInfo(TensorShape(33U, 13U, 2U), 1, DataType::F32);

    auto src1 = create_tensor<CLTensor>(tensor_info);
    auto src2 = create_tensor<CLTensor>(tensor_info);
    compute_float_target_in_place(src1, src2, true);
    validate(CLAccessor(src1), compute_float_reference(tensor_info));
}

TEST_CASE(ComputeWithSameTensor, framework::DatasetMode::ALL)
{
    const auto tensor_info = TensorInfo(TensorShape(33U, 13U, 2U), 1, DataType::F32);

    auto src1 = create_tensor<CLTensor>(tensor_info);
    auto src2 = create_tensor<CLTensor>(tensor_info);
    compute_float_target_in_place(src1, src2, false);
    validate(CLAccessor(src1), compute_float_reference(tensor_info));
}
TEST_SUITE_END() // InPlace

template <typename T>
using CLPReluLayerFixture = PReluLayerValidationFixture<CLTensor, CLAccessor, CLPReluLayer, T>;

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLPReluLayerFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallShapes(), PReluLayerU8Dataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

template <typename T>
using CLPReluLayerQuantizedFixture = PReluLayerValidationQuantizedFixture<CLTensor, CLAccessor, CLPReluLayer, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLPReluLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                   PReluLayerQASYMM8Dataset),
                                                                                                                   framework::dataset::make("QuantizationInfo", { QuantizationInfo(5.f / 255.f, 20) })),
                                                                                                                   framework::dataset::make("QuantizationInfo", { QuantizationInfo(2.f / 255.f, 10) })),
                                                                                                                   framework::dataset::make("QuantizationInfo", { QuantizationInfo(1.f / 255.f, 5) }))

                      )
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall, CLPReluLayerQuantizedFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                  PReluLayerQASYMM8SIGNEDDataset),
                                                                                                                  framework::dataset::make("QuantizationInfo", { QuantizationInfo(5.f / 127.f, 20) })),
                                                                                                                  framework::dataset::make("QuantizationInfo", { QuantizationInfo(2.f / 127.f, 10) })),
                                                                                                                  framework::dataset::make("QuantizationInfo", { QuantizationInfo(1.f / 127.f, 5) }))

                      )
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLPReluLayerFixture<int16_t>, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), PReluLayerS16Dataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunOneDimensional, CLPReluLayerFixture<int16_t>, framework::DatasetMode::ALL, combine(framework::dataset::make("Shape", TensorShape(1U, 16U)), PReluLayerS16Dataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLPReluLayerFixture<half>, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), PReluLayerFP16Dataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp16, 0.01);
}
TEST_SUITE_END()

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLPReluLayerFixture<float>, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), PReluLayerFP32Dataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp32);
}
template <typename T>
using CLPReluLayerBroadcastFixture = PReluLayerBroadcastValidationFixture<CLTensor, CLAccessor, CLPReluLayer, T>;

FIXTURE_DATA_TEST_CASE(RunSmallBroadcast, CLPReluLayerBroadcastFixture<float>, framework::DatasetMode::ALL, combine(datasets::SmallShapesBroadcast(),
                                                                                                                    PReluLayerFP32Dataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp32);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
