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
#include "arm_compute/runtime/NEON/functions/NEAbsoluteDifference.h"
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
#include "tests/validation/fixtures/AbsoluteDifferenceFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Input data sets **/
const auto AbsoluteDifferenceU8Dataset = combine(combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::U8)), framework::dataset::make("DataType",
                                                 DataType::U8));
const auto AbsoluteDifferenceS16Dataset = combine(combine(framework::dataset::make("DataType", { DataType::U8, DataType::S16 }), framework::dataset::make("DataType", DataType::S16)),
                                                  framework::dataset::make("DataType", DataType::S16));
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(AbsoluteDifference)

template <typename T>
using NEAbsoluteDifferenceFixture = AbsoluteDifferenceValidationFixture<Tensor, Accessor, NEAbsoluteDifference, T>;

TEST_SUITE(U8)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), AbsoluteDifferenceU8Dataset),
               shape, data_type0, data_type1, output_data_type)
{
    // Create tensors
    Tensor ref_src1 = create_tensor<Tensor>(shape, data_type0);
    Tensor ref_src2 = create_tensor<Tensor>(shape, data_type1);
    Tensor dst      = create_tensor<Tensor>(shape, output_data_type);

    // Create and Configure function
    NEAbsoluteDifference abs_diff;
    abs_diff.configure(&ref_src1, &ref_src2, &dst);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(ref_src1.info()->padding(), padding);
    validate(ref_src2.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

FIXTURE_DATA_TEST_CASE(RunSmall, NEAbsoluteDifferenceFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallShapes(), AbsoluteDifferenceU8Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEAbsoluteDifferenceFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), AbsoluteDifferenceU8Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U8

TEST_SUITE(S16)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), AbsoluteDifferenceS16Dataset),
               shape, data_type0, data_type1, output_data_type)
{
    // Create tensors
    Tensor ref_src1 = create_tensor<Tensor>(shape, data_type0);
    Tensor ref_src2 = create_tensor<Tensor>(shape, data_type1);
    Tensor dst      = create_tensor<Tensor>(shape, output_data_type);

    // Create and Configure function
    NEAbsoluteDifference abs_diff;
    abs_diff.configure(&ref_src1, &ref_src2, &dst);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(ref_src1.info()->padding(), padding);
    validate(ref_src2.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

FIXTURE_DATA_TEST_CASE(RunSmall, NEAbsoluteDifferenceFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallShapes(), AbsoluteDifferenceS16Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEAbsoluteDifferenceFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), AbsoluteDifferenceS16Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S16

TEST_SUITE_END() // AbsoluteDifference
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
