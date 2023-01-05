/*
 * Copyright (c) 2022-2023 Arm Limited.
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

#include "arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadSketch.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuAdd.h"

#include "tests/CL/CLAccessor.h"
#include "tests/framework/Fixture.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"

#include "tests/datasets/DynamicFusionDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/validation/fixtures/dynamic_fusion/gpu/cl/ElementwiseBinaryFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
/* Synced with tests/validation/CL/ArithmeticAddition.cpp from the standard interface.
 *
 * Difference | Why the difference
 * No quantized tests  | Not supported yet
 * No in place tests   | Not supported yet
 * No activation tests | Not needed in dynamic fusion interface
 *
 */
TEST_SUITE(CL)
TEST_SUITE(DYNAMIC_FUSION)
TEST_SUITE(ADD)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(
               framework::dataset::make("LhsInfo", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),    // Invalid data type combination
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),    // S16 is valid data type for Add
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S32),    // S32 is valid data type for Add
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),    // Mismatching shapes
                                                        TensorInfo(TensorShape(32U,  1U, 1U), 1, DataType::F32),    // Broadcasting allowed for lhs
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8),    // Unsupported data type QASYMM8
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8_SIGNED),    // Unsupported data type QASYMM8
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(15U, 23U, 3U), 1, DataType::F32),    // Broadcast Y dimension is not allowed
                                                        TensorInfo(TensorShape( 3U,  8U, 9U), 1, DataType::S16),    // Broadcast Z dimension is not allowed
                                                        TensorInfo(TensorShape(32U, 13U, 2U, 2), 1, DataType::F32), // Batching is allowed
                                                      }),
               framework::dataset::make("RhsInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S32),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8),    // Unsupported data type QASYMM8
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::QASYMM8_SIGNED),    // Unsupported data type QASYMM8
                                                       TensorInfo(TensorShape(32U,  1U, 1U), 1, DataType::F32),    // Broadcasting allowed for rhs
                                                       TensorInfo(TensorShape(15U,  1U, 3U), 1, DataType::F32),
                                                       TensorInfo(TensorShape( 3U,  8U, 1U), 1, DataType::S16),
                                                       TensorInfo(TensorShape(32U, 13U, 2U, 2), 1, DataType::F32),
                                                      })),
               framework::dataset::make("Expected", { true, false, true, true, false, true, false, false, true, false, false, true})),
               input1_info, input2_info, expected)
{
    // Create a new workload sketch
    auto              cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
    auto              gpu_ctx        = GpuWorkloadContext{ &cl_compile_ctx };
    GpuWorkloadSketch sketch{ &gpu_ctx };

    // Validate Elementwise Add
    auto          lhs_info         = sketch.create_tensor_info(input1_info);
    auto          rhs_info         = sketch.create_tensor_info(input2_info);

    bool res = bool(GpuAdd::validate_op(sketch, &lhs_info, &rhs_info));
    ARM_COMPUTE_EXPECT(res == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

constexpr AbsoluteTolerance<float> tolerance_f16(0.0001f);  /**< Tolerance value for comparing reference's output against implementation's output for DataType::F16 */
constexpr AbsoluteTolerance<float> tolerance_f32(0.0001f);  /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
constexpr float                    tolerance_num = 0.0001f; /**< Tolerance number */

template <typename T>
using DynamicFusionCLAddFixture = DynamicFusionGpuElementwiseBinaryOneOpValidationFixture<CLTensor, CLAccessor, GpuAdd, T>;

template <typename T>
using DynamicFusionCLAddBroadcastFixture = DynamicFusionGpuElementwiseBinaryBroadcastOneOpValidationFixture<CLTensor, CLAccessor, GpuAdd, T>;

template <typename T>
using DynamicFusionCLAddTwoOpsFixture = DynamicFusionGpuElementwiseBinaryTwoOpsValidationFixture<CLTensor, CLAccessor, GpuAdd, T>;

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmallOneOp,
                       DynamicFusionCLAddFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(framework::dataset::make("ElementwiseOp", { ArithmeticOperation::ADD }),
                                               datasets::SmallShapes()),
                                       framework::dataset::make("DataType", { DataType::F32 })),
                               framework::dataset::make("InPlace", { false })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLargeOneOp,
                       DynamicFusionCLAddFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(framework::dataset::make("ElementwiseOp", { ArithmeticOperation::ADD }),
                                               datasets::LargeShapes()),
                                       framework::dataset::make("DataType", { DataType::F32 })),
                               framework::dataset::make("InPlace", { false })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunSmallBroadcastOneOp,
                       DynamicFusionCLAddBroadcastFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(framework::dataset::make("ElementwiseOp", { ArithmeticOperation::ADD }),
                                               datasets::TemporaryLimitedSmallShapesBroadcast()),
                                       framework::dataset::make("DataType", { DataType::F32 })),
                               framework::dataset::make("InPlace", { false })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLargeBroadcastOneOp,
                       DynamicFusionCLAddBroadcastFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(framework::dataset::make("ElementwiseOp", { ArithmeticOperation::ADD }),
                                               datasets::TemporaryLimitedLargeShapesBroadcast()),
                                       framework::dataset::make("DataType", { DataType::F32 })),
                               framework::dataset::make("InPlace", { false })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunSmallTwoOps,
                       DynamicFusionCLAddTwoOpsFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(framework::dataset::make("ElementwiseOp", { ArithmeticOperation::ADD }),
                                                       datasets::DynamicFusionElementwiseBinaryTwoOpsSmallShapes()),
                                               framework::dataset::make("DataType", { DataType::F32 })),
                                       framework::dataset::make("InPlace", { false })),
                               framework::dataset::make("FuseTwoOps", { true })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // FP32

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmallOneOp,
                       DynamicFusionCLAddFixture<half>,
                       framework::DatasetMode::ALL,
                       combine(combine(combine(framework::dataset::make("ElementwiseOp", { ArithmeticOperation::ADD }),
                                               datasets::SmallShapes()),
                                       framework::dataset::make("DataType", { DataType::F16 })),
                               framework::dataset::make("InPlace", { false })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32, tolerance_num);
}

FIXTURE_DATA_TEST_CASE(RunSmallBroadcastOneOp,
                       DynamicFusionCLAddBroadcastFixture<half>,
                       framework::DatasetMode::ALL,
                       combine(combine(combine(framework::dataset::make("ElementwiseOp", { ArithmeticOperation::ADD }),
                                               datasets::TemporaryLimitedSmallShapesBroadcast()),
                                       framework::dataset::make("DataType", { DataType::F16 })),
                               framework::dataset::make("InPlace", { false })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32, tolerance_num);
}

TEST_SUITE_END() // FP16

TEST_SUITE(S32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       DynamicFusionCLAddFixture<int32_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(framework::dataset::make("ElementwiseOp", { ArithmeticOperation::ADD }),
                                               datasets::SmallShapes()),
                                       framework::dataset::make("DataType", { DataType::S32 })),
                               framework::dataset::make("InPlace", { false })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // S32

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       DynamicFusionCLAddFixture<int16_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(framework::dataset::make("ElementwiseOp", { ArithmeticOperation::ADD }),
                                               datasets::SmallShapes()),
                                       framework::dataset::make("DataType", { DataType::S16 })),
                               framework::dataset::make("InPlace", { false })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       DynamicFusionCLAddFixture<int16_t>,
                       framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(framework::dataset::make("ElementwiseOp", { ArithmeticOperation::ADD }),
                                               datasets::LargeShapes()),
                                       framework::dataset::make("DataType", { DataType::S16 })),
                               framework::dataset::make("InPlace", { false })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // S16

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       DynamicFusionCLAddFixture<uint8_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(framework::dataset::make("ElementwiseOp", { ArithmeticOperation::ADD }),
                                               datasets::SmallShapes()),
                                       framework::dataset::make("DataType", { DataType::U8 })),
                               framework::dataset::make("InPlace", { false })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // U8

TEST_SUITE_END() // ADD
TEST_SUITE_END() // DYNAMIC_FUSION
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
