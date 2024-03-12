/*
 * Copyright (c) 2023 Arm Limited.
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

// TODO: Fix testing of CKW Elementwise Binary (COMPMID-6530)
#ifndef ACL_INTERNAL_TEST_CKW_IN_DF

#include "arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadSketch.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuMul.h"

#include "tests/CL/CLAccessor.h"
#include "tests/framework/Fixture.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"

#include "tests/datasets/DynamicFusionDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/validation/fixtures/dynamic_fusion/operators/MulFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
/* Synced with tests/validation/CL/PixelwiseMultiplication.cpp from the standard interface.
 *
 * Difference              | Why the difference
 * No integer tests        | Not supported yet
 * No quantized tests      | Not supported yet
 * No convert policy tests | Not needed as convert policy is ignored by floating types
 * No scale tests          | Not supported yet
 * No rounding modes tests | Not supported yet
 * No in place tests       | Not supported yet
 * No activation tests     | Not needed in dynamic fusion interface
 *
 */
namespace
{
constexpr AbsoluteTolerance<float> tolerance_f16(0.0001f); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F16 */
constexpr AbsoluteTolerance<float> tolerance_f32(0.0001f); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
} // namespace
TEST_SUITE(CL)
TEST_SUITE(DYNAMIC_FUSION)
TEST_SUITE(MUL)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(
               framework::dataset::make("LhsInfo", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F16),
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),    // Invalid data type combination
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),     // Unsupported data type U8
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S8),     // Unsupported data type S8
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),    // Unsupported data type S16
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S32),    // Unsupported data type S32
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8),    // Unsupported data type QASYMM8
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8_SIGNED),    // Unsupported data type QASYMM8_SIGNED
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),    // Mismatching shapes
                                                        TensorInfo(TensorShape(32U,  1U, 1U), 1, DataType::F32),    // Broadcasting allowed for lhs
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(15U, 23U, 3U), 1, DataType::F32),    // Broadcast Y dimension is not allowed
                                                        TensorInfo(TensorShape( 3U,  8U, 9U), 1, DataType::F32),    // Broadcast Z dimension is not allowed
                                                        TensorInfo(TensorShape(32U, 13U, 2U, 2), 1, DataType::F32), // Batching is allowed
                                                      }),
               framework::dataset::make("RhsInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8_SIGNED),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U,  1U, 1U), 1, DataType::F32),    // Broadcasting allowed for rhs
                                                       TensorInfo(TensorShape(15U,  1U, 3U), 1, DataType::F32),
                                                       TensorInfo(TensorShape( 3U,  8U, 1U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U, 2), 1, DataType::F32),
                                                      })),
               framework::dataset::make("Expected", { true, true, false, false, false, false, false, false, false, false, true, true, false, false, true })),
               input1_info, input2_info, expected)
{
    // Create a new workload sketch
    auto              cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
    auto              context        = GpuWorkloadContext{ &cl_compile_ctx };
    GpuWorkloadSketch sketch{ &context };

    // Validate Elementwise Mul
    auto          lhs_info         = context.create_tensor_info(input1_info);
    auto          rhs_info         = context.create_tensor_info(input2_info);

    bool res = bool(GpuMul::validate_op(sketch, &lhs_info, &rhs_info));
    ARM_COMPUTE_EXPECT(res == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using DynamicFusionCLMulFixture = DynamicFusionMulOneOpValidationFixture<CLTensor, CLAccessor, GpuMul, T>;
template <typename T>
using DynamicFusionCLMulBroadcastFixture = DynamicFusionMulBroadcastValidationFixture<CLTensor, CLAccessor, GpuMul, T>;
template <typename T>
using DynamicFusionCLMulTwoOpsFixture = DynamicFusionMulTwoOpsValidationFixture<CLTensor, CLAccessor, GpuMul, T>;

TEST_SUITE(F16)
FIXTURE_DATA_TEST_CASE(RunSmallOneOp,
                       DynamicFusionCLMulFixture<half>,
                       framework::DatasetMode::ALL,
                       combine(combine(datasets::SmallShapes(),
                                       framework::dataset::make("DataType", { DataType::F16 })),
                               framework::dataset::make("InPlace", { false })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunSmallBroadcastOneOp,
                       DynamicFusionCLMulBroadcastFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(datasets::TemporaryLimitedSmallShapesBroadcast(),
                                       framework::dataset::make("DataType", { DataType::F16 })),
                               framework::dataset::make("InPlace", { false })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunLargeBroadcastOneOp,
                       DynamicFusionCLMulBroadcastFixture<half>,
                       framework::DatasetMode::NIGHTLY,
                       combine(combine(datasets::TemporaryLimitedLargeShapesBroadcast(),
                                       framework::dataset::make("DataType", { DataType::F16 })),
                               framework::dataset::make("InPlace", { false })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END() // F16

TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(RunSmallOneOp,
                       DynamicFusionCLMulFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(datasets::SmallShapes(),
                                       framework::dataset::make("DataType", { DataType::F32 })),
                               framework::dataset::make("InPlace", { false })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLargeOneOp,
                       DynamicFusionCLMulFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(combine(datasets::LargeShapes(),
                                       framework::dataset::make("DataType", { DataType::F32 })),
                               framework::dataset::make("InPlace", { false })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunSmallBroadcastOneOp,
                       DynamicFusionCLMulBroadcastFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(datasets::TemporaryLimitedSmallShapesBroadcast(),
                                       framework::dataset::make("DataType", { DataType::F32 })),
                               framework::dataset::make("InPlace", { false })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLargeBroadcastOneOp,
                       DynamicFusionCLMulBroadcastFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(combine(datasets::TemporaryLimitedLargeShapesBroadcast(),
                                       framework::dataset::make("DataType", { DataType::F32 })),
                               framework::dataset::make("InPlace", { false })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunSmallTwoOps,
                       DynamicFusionCLMulTwoOpsFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::DynamicFusionElementwiseBinaryTwoOpsSmallShapes(),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                       framework::dataset::make("InPlace", { false })),
                               framework::dataset::make("FuseTwoOps", { true })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // F32

TEST_SUITE_END() // MUL
TEST_SUITE_END() // DYNAMIC_FUSION
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_INTERNAL_TEST_CKW_IN_DF
