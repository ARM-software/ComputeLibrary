/*
 * Copyright (c) 2023-2024 Arm Limited.
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
#ifndef ACL_INTERNAL_TEST_CKW_IN_DF // Do not include this test if ACL_INTERNAL_TEST_CKW_IN_DF and the op has not been ported to ckw
#include "tests/CL/CLAccessor.h"
#include "tests/datasets/ReshapeLayerDataset.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/validation/fixtures/dynamic_fusion/operators/ReshapeFixture.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(DYNAMIC_FUSION)
TEST_SUITE(RESHAPE)

DATA_TEST_CASE(Validate,
               framework::DatasetMode::ALL,
               zip(zip(framework::dataset::make(
                           "InputInfo",
                           {
                               TensorInfo(TensorShape(9U, 5U, 7U, 3U), 1, DataType::F32),
                               TensorInfo(TensorShape(8U, 4U, 6U, 4U), 1, DataType::F32),
                               TensorInfo(TensorShape(8U, 4U, 6U, 4U), 1, DataType::F32) /*mismatching dimensions*/,
                           }),
                       framework::dataset::make("OutputShape",
                                                {
                                                    TensorShape(9U, 5U, 21U),
                                                    TensorShape(8U, 24U, 4U),
                                                    TensorShape(192U, 192U),
                                                })),
                   framework::dataset::make("Expected", {true, true, false})),
               input_info,
               output_shape,
               expected)
{
    // Create a new workload sketch
    auto              cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
    auto              context        = GpuWorkloadContext{&cl_compile_ctx};
    GpuWorkloadSketch sketch{&context};

    // Create sketch tensors
    TensorShape input_shape = input_info.tensor_shape();
    ARM_COMPUTE_UNUSED(input_shape);
    ITensorInfo *src_info = context.create_tensor_info(input_info);

    ReshapeAttributes attributes;
    attributes.shape(output_shape);
    Status status = GpuReshape::validate_op(sketch, src_info, attributes);
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}

template <typename T>
using DynamicFusionGpuReshapeLayerFixture =
    DynamicFusionGpuReshapeLayerValidationFixture<CLTensor, CLAccessor, GpuReshape, T>;

TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       DynamicFusionGpuReshapeLayerFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallReshapeLayerDataset(),
                               framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // F32

TEST_SUITE(F16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       DynamicFusionGpuReshapeLayerFixture<half>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallReshapeLayerDataset(),
                               framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // F16

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       DynamicFusionGpuReshapeLayerFixture<uint8_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallReshapeLayerDataset(),
                               framework::dataset::make("DataType", DataType::U8)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // U8

TEST_SUITE(S8)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       DynamicFusionGpuReshapeLayerFixture<int8_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallReshapeLayerDataset(),
                               framework::dataset::make("DataType", DataType::S8)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // S8

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       DynamicFusionGpuReshapeLayerFixture<int16_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallReshapeLayerDataset(),
                               framework::dataset::make("DataType", DataType::S16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // S16

TEST_SUITE_END() // RESHAPE
TEST_SUITE_END() // DYNAMIC_FUSION
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute

#endif // ACL_INTERNAL_TEST_CKW_IN_DF
