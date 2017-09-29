/*
 * Copyright (c) 2017 ARM Limited.
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
 * OUT OF OR IN CONCLCTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLArithmeticSubtraction.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ConvertPolicyDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ArithmeticSubtractionFixture.h"

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
const auto ArithmeticSubtractionQS8Dataset = combine(combine(framework::dataset::make("DataType", DataType::QS8), framework::dataset::make("DataType", DataType::QS8)),
                                                     framework::dataset::make("DataType", DataType::QS8));
const auto ArithmeticSubtractionQS16Dataset = combine(combine(framework::dataset::make("DataType", DataType::QS16), framework::dataset::make("DataType", DataType::QS16)),
                                                      framework::dataset::make("DataType", DataType::QS16));
const auto ArithmeticSubtractionFP16Dataset = combine(combine(framework::dataset::make("DataType", DataType::F16), framework::dataset::make("DataType", DataType::F16)),
                                                      framework::dataset::make("DataType", DataType::F16));
const auto ArithmeticSubtractionFP32Dataset = combine(combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::F32)),
                                                      framework::dataset::make("DataType", DataType::F32));
} // namespace

TEST_SUITE(CL)
TEST_SUITE(ArithmeticSubtraction)

template <typename T>
using CLArithmeticSubtractionFixture = ArithmeticSubtractionValidationFixture<CLTensor, CLAccessor, CLArithmeticSubtraction, T>;

TEST_SUITE(U8)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(framework::dataset::concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
               shape, policy)
{
    // Create tensors
    CLTensor ref_src1 = create_tensor<CLTensor>(shape, DataType::U8);
    CLTensor ref_src2 = create_tensor<CLTensor>(shape, DataType::U8);
    CLTensor dst      = create_tensor<CLTensor>(shape, DataType::U8);

    // Create and Configure function
    CLArithmeticSubtraction sub;
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

FIXTURE_DATA_TEST_CASE(RunSmall, CLArithmeticSubtractionFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), ArithmeticSubtractionU8Dataset),
                                                                                                                     framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(S16)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(framework::dataset::concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("DataType", { DataType::U8, DataType::S16 })),
                                                                   framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
               shape, data_type, policy)
{
    // Create tensors
    CLTensor ref_src1 = create_tensor<CLTensor>(shape, data_type);
    CLTensor ref_src2 = create_tensor<CLTensor>(shape, DataType::S16);
    CLTensor dst      = create_tensor<CLTensor>(shape, DataType::S16);

    // Create and Configure function
    CLArithmeticSubtraction sub;
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

FIXTURE_DATA_TEST_CASE(RunSmall, CLArithmeticSubtractionFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), ArithmeticSubtractionS16Dataset),
                                                                                                                     framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLArithmeticSubtractionFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), ArithmeticSubtractionS16Dataset),
                                                                                                                   framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

template <typename T>
using CLArithmeticSubtractionFixedPointFixture = ArithmeticSubtractionValidationFixedPointFixture<CLTensor, CLAccessor, CLArithmeticSubtraction, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QS8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLArithmeticSubtractionFixedPointFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(), ArithmeticSubtractionQS8Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                       framework::dataset::make("FractionalBits", 1, 7)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLArithmeticSubtractionFixedPointFixture<int8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), ArithmeticSubtractionQS8Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                       framework::dataset::make("FractionalBits", 1, 7)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(QS16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLArithmeticSubtractionFixedPointFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallShapes(),
                       ArithmeticSubtractionQS16Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                       framework::dataset::make("FractionalBits", 1, 15)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLArithmeticSubtractionFixedPointFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), ArithmeticSubtractionQS16Dataset),
                       framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
                       framework::dataset::make("FractionalBits", 1, 15)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLArithmeticSubtractionFixture<half>, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), ArithmeticSubtractionFP16Dataset),
                                                                                                            framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(FP32)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(framework::dataset::concat(datasets::SmallShapes(), datasets::LargeShapes()), framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })),
               shape, policy)
{
    // Create tensors
    CLTensor ref_src1 = create_tensor<CLTensor>(shape, DataType::F32);
    CLTensor ref_src2 = create_tensor<CLTensor>(shape, DataType::F32);
    CLTensor dst      = create_tensor<CLTensor>(shape, DataType::F32);

    // Create and Configure function
    CLArithmeticSubtraction sub;
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

FIXTURE_DATA_TEST_CASE(RunSmall, CLArithmeticSubtractionFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), ArithmeticSubtractionFP32Dataset),
                                                                                                                   framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLArithmeticSubtractionFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), ArithmeticSubtractionFP32Dataset),
                                                                                                                 framework::dataset::make("ConvertPolicy", { ConvertPolicy::SATURATE, ConvertPolicy::WRAP })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
