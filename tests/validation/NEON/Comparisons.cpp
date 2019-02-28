/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEElementwiseOperations.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
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
                                       framework::dataset::make("DataType", { DataType::QASYMM8,
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                                                              DataType::F16,
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
                                                                              DataType::F32
                                                                            }));

const auto run_small_dataset = combine(datasets::ComparisonOperations(), datasets::SmallShapes());
const auto run_large_dataset = combine(datasets::ComparisonOperations(), datasets::LargeShapes());

} // namespace

TEST_SUITE(NEON)
TEST_SUITE(Comparison)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
        framework::dataset::make("Input1Info", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32), // Invalid output type
                                                 TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32), // Mismatching input types
                                                 TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32), // Mismatching shapes
                                                 TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
        }),
        framework::dataset::make("Input2Info",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S32),
                                                TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
        })),
        framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::U8),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
        })),
        framework::dataset::make("Expected", { false, false, false, true})),
        input1_info, input2_info, output_info, expected)
{
    Status s = NEElementwiseComparison::validate(&input1_info.clone()->set_is_resizable(false),
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
    Tensor ref_src1 = create_tensor<Tensor>(shape, data_type);
    Tensor ref_src2 = create_tensor<Tensor>(shape, data_type);
    Tensor dst      = create_tensor<Tensor>(shape, DataType::U8);

    // Create and Configure function
    NEElementwiseComparison compare;
    compare.configure(&ref_src1, &ref_src2, &dst, ComparisonOperation::Equal);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);
}

template <typename T>
using NEComparisonFixture = ComparisonValidationFixture<Tensor, Accessor, NEElementwiseComparison, T>;

TEST_SUITE(Float)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEComparisonFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(run_small_dataset, framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEComparisonFixture<half>,
                       framework::DatasetMode::NIGHTLY,
                       combine(run_large_dataset, framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // FP16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEComparisonFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(run_small_dataset, framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEComparisonFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(run_large_dataset, framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

template <typename T>
using NEComparisonQuantizedFixture = ComparisonValidationQuantizedFixture<Tensor, Accessor, NEElementwiseComparison, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEComparisonQuantizedFixture<uint8_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(run_small_dataset, framework::dataset::make("DataType", DataType::QASYMM8)),
                                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(5.f / 255.f, 20) })),
                               framework::dataset::make("QuantizationInfo", { QuantizationInfo(2.f / 255.f, 10) })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END() // Comparison
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
