/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLComparison.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ComparisonOperationsDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ComparisonFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
const auto configure_dataset = combine(datasets::SmallShapes(),
                                       framework::dataset::make("DataType", { DataType::QASYMM8, DataType::F16, DataType::F32 }));

const auto run_small_dataset = combine(datasets::ComparisonOperations(), datasets::SmallShapes());
const auto run_large_dataset = combine(datasets::ComparisonOperations(), datasets::LargeShapes());

} // namespace

TEST_SUITE(CL)
TEST_SUITE(Comparison)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
        framework::dataset::make("Input1Info", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32), // Invalid output type
                                                 TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32), // Mismatching input types
                                                 TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Window shrink
                                                 TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32), // Mismatching shapes
                                                 TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
        }),
        framework::dataset::make("Input2Info",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S32),
                                                TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
        })),
        framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::U8),
                                                TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::U8),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
        })),
        framework::dataset::make("Expected", { false, false, false, false, true})),
        input1_info, input2_info, output_info, expected)
{
    Status s = CLComparison::validate(&input1_info.clone()->set_is_resizable(false),
                                      &input2_info.clone()->set_is_resizable(false),
                                      &output_info.clone()->set_is_resizable(false),
                                      ComparisonOperation::Equal);
    ARM_COMPUTE_EXPECT(bool(s) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, configure_dataset,
               shape, data_type)
{
    // Create tensors
    CLTensor ref_src1 = create_tensor<CLTensor>(shape, data_type);
    CLTensor ref_src2 = create_tensor<CLTensor>(shape, data_type);
    CLTensor dst      = create_tensor<CLTensor>(shape, DataType::U8);

    // Create and Configure function
    CLComparison compare;
    compare.configure(&ref_src1, &ref_src2, &dst, ComparisonOperation::Equal);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    const int num_elems_processed_per_iteration = 16 / ref_src1.info()->element_size();

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), num_elems_processed_per_iteration).required_padding();
    validate(ref_src1.info()->padding(), padding);
    validate(ref_src2.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

template <typename T>
using CLComparisonFixture = ComparisonValidationFixture<CLTensor, CLAccessor, CLComparison, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       CLComparisonFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(run_small_dataset, framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       CLComparisonFixture<half>,
                       framework::DatasetMode::NIGHTLY,
                       combine(run_large_dataset, framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // FP16

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       CLComparisonFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(run_small_dataset, framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       CLComparisonFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(run_large_dataset, framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

template <typename T>
using CLComparisonQuantizedFixture = ComparisonValidationQuantizedFixture<CLTensor, CLAccessor, CLComparison, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       CLComparisonQuantizedFixture<uint8_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(run_small_dataset, framework::dataset::make("DataType", DataType::QASYMM8)),
                                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(5.f / 255.f, 20) })),
                               framework::dataset::make("QuantizationInfo", { QuantizationInfo(2.f / 255.f, 10) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END() // Comparison
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
