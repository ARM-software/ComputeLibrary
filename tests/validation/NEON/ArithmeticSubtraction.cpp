/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEArithmeticSubtraction.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ConvertPolicyDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ArithmeticOperationsFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Input data sets **/
const auto ArithmeticSubtractionU8Dataset = combine(combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::U8)),
                                                    framework::dataset::make("DataType",
                                                                             DataType::U8));
const auto ArithmeticSubtractionS16Dataset = combine(combine(framework::dataset::make("DataType", { DataType::U8, DataType::S16 }), framework::dataset::make("DataType", DataType::S16)),
                                                     framework::dataset::make("DataType", DataType::S16));
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
const auto ArithmeticSubtractionFP16Dataset = combine(combine(framework::dataset::make("DataType", DataType::F16), framework::dataset::make("DataType", DataType::F16)),
                                                      framework::dataset::make("DataType", DataType::F16));
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
const auto ArithmeticSubtractionFP32Dataset = combine(combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::F32)),
                                                      framework::dataset::make("DataType", DataType::F32));
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(ArithmeticSubtraction)

template <typename T>
using NEArithmeticSubtractionFixture = ArithmeticSubtractionValidationFixture<Tensor, Accessor, NEArithmeticSubtraction, T>;

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
        framework::dataset::make("Input1Info", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                 TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                 TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::U8),      // Window shrink
                                                 TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),      // Invalid data type combination
                                                 TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),     // Mismatching shapes
        }),
        framework::dataset::make("Input2Info",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::U8),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),
                                                TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
        })),
        framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S16),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::U8),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                TensorInfo(TensorShape(48U, 11U, 2U), 1, DataType::F32),
        })),
        framework::dataset::make("Expected", { true, true, false, false, false})),
        input1_info, input2_info, output_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(NEArithmeticSubtraction::validate(&input1_info.clone()->set_is_resizable(false), &input2_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), ConvertPolicy::WRAP)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE(U8)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
               shape, policy)
{
    // Create tensors
    Tensor ref_src1 = create_tensor<Tensor>(shape, DataType::U8);
    Tensor ref_src2 = create_tensor<Tensor>(shape, DataType::U8);
    Tensor dst      = create_tensor<Tensor>(shape, DataType::U8);

    // Create and Configure function
    NEArithmeticSubtraction sub;
    sub.configure(&ref_src1, &ref_src2, &dst, policy);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(ref_src1.info()->padding(), padding);
    validate(ref_src2.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

FIXTURE_DATA_TEST_CASE(RunSmall, NEArithmeticSubtractionFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), ArithmeticSubtractionU8Dataset),
                                                                                                                     framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U8

TEST_SUITE(S16)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType", { DataType::U8, DataType::S16 })),
                                                                   framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
               shape, data_type, policy)
{
    // Create tensors
    Tensor ref_src1 = create_tensor<Tensor>(shape, data_type);
    Tensor ref_src2 = create_tensor<Tensor>(shape, DataType::S16);
    Tensor dst      = create_tensor<Tensor>(shape, DataType::S16);

    // Create and Configure function
    NEArithmeticSubtraction sub;
    sub.configure(&ref_src1, &ref_src2, &dst, policy);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(ref_src1.info()->padding(), padding);
    validate(ref_src2.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

FIXTURE_DATA_TEST_CASE(RunSmall, NEArithmeticSubtractionFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), ArithmeticSubtractionS16Dataset),
                                                                                                                     framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEArithmeticSubtractionFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), ArithmeticSubtractionS16Dataset),
                                                                                                                   framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S16

TEST_SUITE(Float)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(F16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEArithmeticSubtractionFixture<half>, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), ArithmeticSubtractionFP16Dataset),
                                                                                                            framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // F16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(F32)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
               shape, policy)
{
    // Create tensors
    Tensor ref_src1 = create_tensor<Tensor>(shape, DataType::F32);
    Tensor ref_src2 = create_tensor<Tensor>(shape, DataType::F32);
    Tensor dst      = create_tensor<Tensor>(shape, DataType::F32);

    // Create and Configure function
    NEArithmeticSubtraction sub;
    sub.configure(&ref_src1, &ref_src2, &dst, policy);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(ref_src1.info()->padding(), padding);
    validate(ref_src2.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

FIXTURE_DATA_TEST_CASE(RunSmall, NEArithmeticSubtractionFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), ArithmeticSubtractionFP32Dataset),
                                                                                                                   framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEArithmeticSubtractionFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), ArithmeticSubtractionFP32Dataset),
                                                                                                                 framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

template <typename T>
using NEArithmeticSubtractionBroadcastFixture = ArithmeticSubtractionBroadcastValidationFixture<Tensor, Accessor, NEArithmeticSubtraction, T>;

FIXTURE_DATA_TEST_CASE(RunSmallBroadcast, NEArithmeticSubtractionBroadcastFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapesBroadcast(),
                       ArithmeticSubtractionFP32Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLargeBroadcast, NEArithmeticSubtractionBroadcastFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapesBroadcast(),
                       ArithmeticSubtractionFP32Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // F32
TEST_SUITE_END() // Float

TEST_SUITE_END() // ArithmeticSubtraction
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
