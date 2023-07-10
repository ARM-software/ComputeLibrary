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
#include "arm_compute/core/Types.h"
#include "arm_compute/dynamic_fusion/sketch/attributes/ClampAttributes.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuClamp.h"
#include "arm_compute/runtime/CL/CLTensor.h"

#include "tests/CL/CLAccessor.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/dynamic_fusion/operators/ClampFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr float                    epsilon = 1e-6f;
constexpr AbsoluteTolerance<float> tolerance(epsilon);
} // namespace

TEST_SUITE(CL)
TEST_SUITE(DYNAMIC_FUSION)
TEST_SUITE(CLAMP)
// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
                framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F16),
                                                        TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),    // Minimum value larger than maximum value
                                                    }),
                framework::dataset::make("MinVal", { 0.2f,
                                                     1.5f,
                                                     9.0f,
                                                    })),
                framework::dataset::make("MaxVal", { 0.5f,
                                                     2.0f,
                                                     1.0f,
                                                    })),
                framework::dataset::make("Expected", { true, true, false })),
                input_info, min_val, max_val, expected)
{
    // Create a new workload sketch
    CLCompileContext cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
    GpuWorkloadContext context{ &cl_compile_ctx };
    GpuWorkloadSketch sketch{ &context };

    // Fuse Clamp
    const TensorInfo src_info = context.create_tensor_info(input_info);

    ClampAttributes attributes {};
    attributes.min_val(min_val)
              .max_val(max_val);

    const bool res = static_cast<bool>(GpuClamp::validate_op(sketch, &src_info, attributes));
    ARM_COMPUTE_EXPECT(res == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using DynamicFusionClampOpFixture = DynamicFusionClampValidationFixture<CLTensor, CLAccessor, GpuClamp, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmallOneOp,
                       DynamicFusionClampOpFixture<half>,
                       framework::DatasetMode::ALL,
                       combine(combine(combine(datasets::SmallShapes(),
                                               framework::dataset::make("ClampAttributes", { ClampAttributes().min_val(0.1f).max_val(0.6f) })),
                                       framework::dataset::make("Fuse", { false })),
                               framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance);
}

FIXTURE_DATA_TEST_CASE(RunSmall5dOneOp,
                       DynamicFusionClampOpFixture<half>,
                       framework::DatasetMode::ALL,
                       combine(combine(combine(datasets::Small5dShapes(),
                                               framework::dataset::make("ClampAttributes", { ClampAttributes().min_val(0.1f).max_val(0.6f) })),
                                       framework::dataset::make("Fuse", { false })),
                               framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    ARM_COMPUTE_TEST_INFO("Currently 5D+ tensors are unsupported for this operation.");
    framework::ARM_COMPUTE_PRINT_INFO();
}

FIXTURE_DATA_TEST_CASE(RunSmallTwoOps,
                       DynamicFusionClampOpFixture<half>,
                       framework::DatasetMode::ALL,
                       combine(combine(combine(datasets::SmallShapes(),
                                               framework::dataset::make("ClampAttributes", { ClampAttributes().min_val(0.2f).max_val(0.4f) })),
                                       framework::dataset::make("Fuse", { true })),
                               framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance);
}

TEST_SUITE_END() // FP16

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmallOneOp,
                       DynamicFusionClampOpFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(combine(combine(datasets::SmallShapes(),
                                               framework::dataset::make("ClampAttributes", { ClampAttributes().min_val(0.3f).max_val(0.7f) })),
                                       framework::dataset::make("Fuse", { false })),
                               framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance);
}

FIXTURE_DATA_TEST_CASE(RunSmall5dOneOp,
                       DynamicFusionClampOpFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(combine(combine(datasets::Small5dShapes(),
                                               framework::dataset::make("ClampAttributes", { ClampAttributes().min_val(0.3f).max_val(0.7f) })),
                                       framework::dataset::make("Fuse", { false })),
                               framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    ARM_COMPUTE_TEST_INFO("Currently 5D+ tensors are unsupported for this operation.");
    framework::ARM_COMPUTE_PRINT_INFO();
}

FIXTURE_DATA_TEST_CASE(RunSmallTwoOps,
                       DynamicFusionClampOpFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(combine(combine(datasets::SmallShapes(),
                                               framework::dataset::make("ClampAttributes", { ClampAttributes().min_val(0.1f).max_val(0.9f) })),
                                       framework::dataset::make("Fuse", { true })),
                               framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance);
}

TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

TEST_SUITE_END() // CLAMP
TEST_SUITE_END() // DYNAMIC_FUSION
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
