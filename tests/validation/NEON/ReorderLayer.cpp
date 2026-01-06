/*
 * Copyright (c) 2023-2026 Arm Limited.
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
#if defined(__aarch64__)

#include "arm_compute/runtime/NEON/functions/NEReorderLayer.h"
#include "arm_compute/runtime/Tensor.h"

#include "src/core/NEON/kernels/arm_gemm/utils.hpp"
#include "src/core/NEON/kernels/NEReorderKernel.h"
#include "tests/datasets/ReorderLayerDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/validation/fixtures/ReorderFixture.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
using framework::dataset::make;

TEST_SUITE(NEON)
TEST_SUITE(ReorderLayer)

template <typename TOut>
using NEReorderLayerAlias = ReorderValidationFixture<Tensor, Accessor, NEReorderLayer, TOut, float>;

TEST_SUITE(FP32)
#if defined(ARM_COMPUTE_ENABLE_SVE)
DATA_TEST_CASE(ValidateReorderOHWIo8,
               framework::DatasetMode::ALL,
               combine(zip(make("InShape", {TensorShape(10U, 9U), TensorShape(234U, 301U)}),
                           make("OutShape", {TensorShape(10U, 16U), TensorShape(234U, 304U)})),
                       zip(make("InputWeightFormat", {WeightFormat::OHWI}),
                           make("OutputWeightFormat", {WeightFormat::OHWIo8}))),
               input_shape,
               output_shape,
               input_wf,
               output_wf)
{
    arm_compute::NEReorderLayer reorder_layer;
    bool                        expected_bool_status = true;
    TensorInfo                  input_tensor_info(input_shape, 1, DataType::F32);
    TensorInfo                  output_tensor_info(output_shape, 1, DataType::F32);

    Status status =
        reorder_layer.validate(&input_tensor_info, &output_tensor_info, input_wf, output_wf, true /* transpose */);

    ARM_COMPUTE_EXPECT((expected_bool_status == bool(status)), framework::LogLevel::ERRORS);
}

FIXTURE_DATA_TEST_CASE(RunInterleave8,
                       NEReorderLayerAlias<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::ReorderLayerDatasetInterleave8(),
                               make("OutputWeightFormat", WeightFormat::OHWIo8),
                               make("InputDataType", DataType::F32),
                               make("OutputDataType", DataType::F32),
                               make("Transpose", {true, false})))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
#endif // ARM_COMPUTE_ENABLE_SVE

FIXTURE_DATA_TEST_CASE(RunInterleave4,
                       NEReorderLayerAlias<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::ReorderLayerDatasetInterleave4(),
                               make("OutputWeightFormat", WeightFormat::OHWIo4),
                               make("InputDataType", DataType::F32),
                               make("OutputDataType", DataType::F32),
                               make("Transpose", {true, false})))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

TEST_SUITE_END() // FP32

#ifdef ARM_COMPUTE_ENABLE_BF16
TEST_SUITE(BF16)
FIXTURE_DATA_TEST_CASE(RunInterleave4Block4,
                       NEReorderLayerAlias<bfloat16>,
                       framework::DatasetMode::ALL,
                       combine(datasets::ReorderLayerDatasetInterleave4Block4(),
                               make("OutputWeightFormat", WeightFormat::OHWIo4i4),
                               make("InputDataType", DataType::F32),
                               make("OutputDataType", DataType::BFLOAT16),
                               make("Transpose", {true, false})))
{
    if (CPUInfo::get().has_bf16())
    {
        // Validate output
        validate(Accessor(_target), _reference);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support bf16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}

#ifdef ARM_COMPUTE_ENABLE_SVE
FIXTURE_DATA_TEST_CASE(RunInterleave8Block4,
                       NEReorderLayerAlias<bfloat16>,
                       framework::DatasetMode::ALL,
                       combine(datasets::ReorderLayerDatasetInterleave8Block4(),
                               make("OutputWeightFormat", WeightFormat::OHWIo8i4),
                               make("InputDataType", DataType::F32),
                               make("OutputDataType", DataType::BFLOAT16),
                               make("Transpose", {true, false})))
{
    if (CPUInfo::get().has_bf16())
    {
        // Validate output
        validate(Accessor(_target), _reference);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support bf16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
#endif // ARM_COMPUTE_ENABLE_SVE

TEST_SUITE_END() // BF16
#endif           // ARM_COMPUTE_ENABLE_BF16

TEST_SUITE_END() // ReorderLayer
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute

#endif // defined(__aarch64__)
