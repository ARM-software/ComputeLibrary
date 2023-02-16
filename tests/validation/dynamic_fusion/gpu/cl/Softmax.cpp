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
#include "arm_compute/core/Types.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuSoftmax.h"

#include "tests/CL/CLAccessor.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/dynamic_fusion/operators/SoftmaxFixture.h"

using namespace arm_compute::experimental::dynamic_fusion;

namespace arm_compute
{
namespace test
{
namespace validation
{
/** Tolerance for float operations */
RelativeTolerance<half>  tolerance_f16(half(0.2));
RelativeTolerance<float> tolerance_f32(0.001f);

TEST_SUITE(CL)
TEST_SUITE(DYNAMIC_FUSION)
TEST_SUITE(SOFTMAX)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 13U), 1, DataType::F32), // Mismatching data types
                                                       TensorInfo(TensorShape(27U, 13U), 1, DataType::F32), // Mismatching shapes
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::S32), // Unsupported data type
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),

                                                      }),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(27U, 13U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(27U, 11U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::QASYMM16), // Unsupported data type
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U), 1, DataType::F32),

                                                     })),
               framework::dataset::make("beta", { 1.0,
                                                  2.0,
                                                  2.0,
                                                  1.0,
                                                  1.0,
                                                  1.0,
                                                  1.0,
                                                  1.0,
                                                  1.0,
                                                  1.0,
                                                })),
               framework::dataset::make("axis", {
                                                  0,
                                                  0,
                                                  1,  // Invalid as axis != 0
                                                  0,
                                                  0,
                                                  0,
                                                  -3, // Invalid as axis != 0
                                                  2,  // Invalid as axis != 0
                                                  1,  // Invalid as axis != 0
                                                  -1, // Invalid as axis != 0
                                                })),
               framework::dataset::make("Expected", { false, false, false, true, false, false, false, false, false, false})),
               input_info, output_info, beta, axis, expected)
{
    // Create a new workload sketch
    CLCompileContext   cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
    GpuWorkloadContext gpu_ctx        = GpuWorkloadContext{ &cl_compile_ctx };
    GpuWorkloadSketch  sketch{ &gpu_ctx };

    SoftmaxAttributes softmax_attr{};
    softmax_attr.axis(axis).beta(beta).is_log_softmax(false);
    TensorInfo src_info  = sketch.create_tensor_info(input_info);
    TensorInfo dst_info = sketch.create_tensor_info(output_info);
    const bool res = static_cast<bool>(GpuSoftmax::validate_op(sketch, &src_info, &dst_info, softmax_attr));
    ARM_COMPUTE_EXPECT(res == expected, framework::LogLevel::ERRORS);
}

template <typename T>
using DynamicFusionSoftmaxLayerFixture = DynamicFusionSoftmaxValidationFixture<CLTensor, CLAccessor, GpuSoftmax, T>;

TEST_SUITE(FLOAT)
TEST_SUITE(FP32)

FIXTURE_DATA_TEST_CASE(RunSmall, DynamicFusionSoftmaxLayerFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SoftmaxLayerSmallShapes(),
                                                                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                                                           framework::dataset::make("Beta", { 1.0f, 2.0f })),
                                                                                                   framework::dataset::make("Axis", { 0 })),
                                                                                                   framework::dataset::make("is_log", {false, true})))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}


FIXTURE_DATA_TEST_CASE(RunLarge, DynamicFusionSoftmaxLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::SoftmaxLayerLargeShapes(),
                                                                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                                                           framework::dataset::make("Beta", { 1.0f, 2.0f })),
                                                                                                   framework::dataset::make("Axis", { 0 })),
                                                                                                   framework::dataset::make("is_log", {false, true})))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}


FIXTURE_DATA_TEST_CASE(Run4D, DynamicFusionSoftmaxLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::SoftmaxLayer4DShapes(),
                                                                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                                                           framework::dataset::make("Beta", { 1.0f, 2.0f })),
                                                                                                   framework::dataset::make("Axis", { 0 })),
                                                                                                   framework::dataset::make("is_log", {false, true})))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // FP32
TEST_SUITE(FP16)

FIXTURE_DATA_TEST_CASE(RunSmall, DynamicFusionSoftmaxLayerFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SoftmaxLayerSmallShapes(),
                                                                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                                                           framework::dataset::make("Beta", { 1.0f, 2.0f })),
                                                                                                   framework::dataset::make("Axis", { 0 })),
                                                                                                   framework::dataset::make("is_log", {false, true})))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}


FIXTURE_DATA_TEST_CASE(RunLarge, DynamicFusionSoftmaxLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::SoftmaxLayerLargeShapes(),
                                                                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                                                           framework::dataset::make("Beta", { 1.0f, 2.0f })),
                                                                                                   framework::dataset::make("Axis", { 0 })),
                                                                                                   framework::dataset::make("is_log", {false, true})))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}


FIXTURE_DATA_TEST_CASE(Run4D, DynamicFusionSoftmaxLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::SoftmaxLayer4DShapes(),
                                                                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                                                           framework::dataset::make("Beta", { 1.0f, 2.0f })),
                                                                                                   framework::dataset::make("Axis", { 0 })),
                                                                                                   framework::dataset::make("is_log", {false, true})))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END() // FP16
TEST_SUITE_END() // FLOAT

TEST_SUITE_END() // SOFTMAX
TEST_SUITE_END() // DYNAMIC_FUSION
TEST_SUITE_END() // CL

} // namespace validation
} // namespace test
} // namespace arm_compute
