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
RelativeTolerance<float> tolerance_fp32(0.001f);
/** Input data sets **/
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
RelativeTolerance<half> tolerance_fp16(static_cast<half>(0.01f));
const auto              ElementwisePowerFP16Dataset = combine(combine(framework::dataset::make("DataType", DataType::F16), framework::dataset::make("DataType", DataType::F16)),
                                                              framework::dataset::make("DataType", DataType::F16));
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
const auto ElementwisePowerFP32Dataset = combine(combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::F32)),
                                                 framework::dataset::make("DataType", DataType::F32));
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(ElementwisePower)

template <typename T>
using NEElementwisePowerFixture = ElementwisePowerValidationFixture<Tensor, Accessor, NEElementwisePower, T>;

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
               framework::dataset::make("Input1Info", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),     // Invalid data type combination
                                                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),     // Mismatching shapes
                                                      }),
               framework::dataset::make("Input2Info",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S32),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("Expected", { true, true, true, false, false})),
               input1_info, input2_info, output_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(NEElementwisePower::validate(&input1_info.clone()->set_is_resizable(false), &input2_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false))) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE(Float)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(F16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEElementwisePowerFixture<half>, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), ElementwisePowerFP16Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp16, 0.01);
}
TEST_SUITE_END() // F16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(F32)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, framework::dataset::concat(datasets::SmallShapes(), datasets::LargeShapes()),
               shape)
{
    // Create tensors
    Tensor ref_src1 = create_tensor<Tensor>(shape, DataType::F32);
    Tensor ref_src2 = create_tensor<Tensor>(shape, DataType::F32);
    Tensor dst      = create_tensor<Tensor>(shape, DataType::F32);

    // Create and Configure function
    NEElementwisePower power;
    power.configure(&ref_src1, &ref_src2, &dst);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);
}

FIXTURE_DATA_TEST_CASE(RunSmall, NEElementwisePowerFixture<float>, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), ElementwisePowerFP32Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32, 0.01);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEElementwisePowerFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), ElementwisePowerFP32Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32, 0.01);
}

template <typename T>
using NEElementwisePowerBroadcastFixture = ElementwisePowerBroadcastValidationFixture<Tensor, Accessor, NEElementwisePower, T>;

FIXTURE_DATA_TEST_CASE(RunSmallBroadcast, NEElementwisePowerBroadcastFixture<float>, framework::DatasetMode::ALL, combine(datasets::SmallShapesBroadcast(),
                       ElementwisePowerFP32Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32, 0.01);
}

FIXTURE_DATA_TEST_CASE(RunLargeBroadcast, NEElementwisePowerBroadcastFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapesBroadcast(),
                       ElementwisePowerFP32Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32, 0.01);
}
TEST_SUITE_END() // F32
TEST_SUITE_END() // Float

TEST_SUITE_END() // ElementwisePower
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
