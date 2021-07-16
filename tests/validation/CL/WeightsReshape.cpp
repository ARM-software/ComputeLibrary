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
#include "src/core/gpu/cl/kernels/ClWeightsReshapeKernel.h"
#include "tests/CL/CLAccessor.h"
#include "tests/CL/Helper.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/WeightsReshapeFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(WeightsReshape)

using ClWeightsReshape = ClSynthetizeOperatorWithBorder<opencl::kernels::ClWeightsReshapeKernel>;

/** Validate tests
 *
 * A series of validation tests on configurations which according to the API specification
 * the function should fail against.
 *
 * Checks performed in order:
 *     - Mismachting data type: bias need to has same data type as input
 *     - Mismachting data type: output need to has same data type as input
 *     - Bias only supports FP32/FP16
 *     - num_groups != 1 is only supported for NCHW data layout
 *     - Bias' shape need to match input's shape.
 */
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
                                                                      framework::dataset::make("InputInfo",
{
    TensorInfo(TensorShape(3U, 3U, 2U, 4U), 1, DataType::F32),                   // Mismatching data type
    TensorInfo(TensorShape(3U, 3U, 2U, 4U), 1, DataType::F32),                   // Mismatching data type
    TensorInfo(TensorShape(3U, 3U, 2U, 4U), 1, DataType::QASYMM8),               // Bias only supports FP32/FP16
    TensorInfo(TensorShape(3U, 3U, 2U, 4U), 1, DataType::F32, DataLayout::NHWC), // num_groups != 1 is only supported for NCHW data layout
    TensorInfo(TensorShape(3U, 3U, 2U, 4U, 4U), 1, DataType::F32),               // Bias' shape need to match input's shape
    TensorInfo(TensorShape(3U, 3U, 2U, 4U, 4U), 1, DataType::F32),               // Bias' shape need to match input's shape
}),
framework::dataset::make("BiasesInfo",
{
    TensorInfo(TensorShape(4U), 1, DataType::F16),
    TensorInfo(TensorShape(4U), 1, DataType::F32),
    TensorInfo(TensorShape(4U), 1, DataType::QASYMM8),
    TensorInfo(TensorShape(4U), 1, DataType::F32),
    TensorInfo(TensorShape(4U, 3U), 1, DataType::F32),
    TensorInfo(TensorShape(3U, 4U), 1, DataType::F32),
})),
framework::dataset::make("OutputInfo",
{
    TensorInfo(TensorShape(4U, 19U), 1, DataType::F32),
    TensorInfo(TensorShape(4U, 19U), 1, DataType::F16),
    TensorInfo(TensorShape(4U, 19U), 1, DataType::QASYMM8),
    TensorInfo(TensorShape(4U, 19U), 1, DataType::F32),
    TensorInfo(TensorShape(4U, 19U), 1, DataType::F32),
    TensorInfo(TensorShape(4U, 19U), 1, DataType::F32),
})),
framework::dataset::make("NumGroups", { 1, 1, 1, 2, 1, 2 })),
framework::dataset::make("Expected", { false, false, false, false, false, false })),
input_info, biases_info, output_info, num_groups, expected)
{
    bool status = bool(opencl::kernels::ClWeightsReshapeKernel::validate(&input_info, &biases_info, &output_info, num_groups));
    ARM_COMPUTE_EXPECT(status == expected, framework::LogLevel::ERRORS);
}

template <typename T>
using ClWeightsReshapeFixture = WeightsReshapeOpValidationFixture<CLTensor, CLAccessor, ClWeightsReshape, T>;

TEST_SUITE(Float)
FIXTURE_DATA_TEST_CASE(FP32, ClWeightsReshapeFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(framework::dataset::make("InputShape", { TensorShape(3U, 3U, 48U, 120U) }),
                                                                                                                  framework::dataset::make("DataType", DataType::F32)),
                                                                                                          framework::dataset::make("HasBias", { true, false })),
                                                                                                  framework::dataset::make("NumGroups", { 1, 2 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(FP16, ClWeightsReshapeFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(framework::dataset::make("InputShape", { TensorShape(13U, 13U, 96U, 240U) }),
                                                                                                                 framework::dataset::make("DataType", DataType::F16)),
                                                                                                         framework::dataset::make("HasBias", { true, false })),
                                                                                                 framework::dataset::make("NumGroups", { 3, 4 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(BFloat16, ClWeightsReshapeFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(framework::dataset::make("InputShape", { TensorShape(9U, 9U, 96U, 240U) }),
                                                                                                                     framework::dataset::make("DataType", DataType::BFLOAT16)),
                                                                                                             framework::dataset::make("HasBias", { false })),
                                                                                                     framework::dataset::make("NumGroups", { 3, 4 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

TEST_SUITE_END()

TEST_SUITE(Quantized)
FIXTURE_DATA_TEST_CASE(QASYMM8, ClWeightsReshapeFixture<uint8_t>, framework::DatasetMode::ALL, combine(combine(combine(framework::dataset::make("InputShape", { TensorShape(5U, 5U, 48U, 120U) }),
                                                                                                                       framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                               framework::dataset::make("HasBias", { false })),
                                                                                                       framework::dataset::make("NumGroups", { 1, 2 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(QASYMM8_SIGNED, ClWeightsReshapeFixture<uint8_t>, framework::DatasetMode::ALL, combine(combine(combine(framework::dataset::make("InputShape", { TensorShape(5U, 5U, 48U, 120U) }),
                                                                                                                      framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                                                                                                                      framework::dataset::make("HasBias", { false })),
                                                                                                              framework::dataset::make("NumGroups", { 1, 2 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
