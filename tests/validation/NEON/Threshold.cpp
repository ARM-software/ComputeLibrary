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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/runtime/NEON/functions/NEThreshold.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/datasets/ThresholdDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ThresholdFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(NEON)
TEST_SUITE(Threshold)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()), datasets::MixedThresholdDataset()),
                                                                   framework::dataset::make("DataType", DataType::U8)),
               shape, threshold, false_value, true_value, type, upper, data_type)
{
    // Create tensors
    Tensor src = create_tensor<Tensor>(shape, data_type);
    Tensor dst = create_tensor<Tensor>(shape, data_type);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    NEThreshold thrsh;
    thrsh.configure(&src, &dst, threshold, false_value, true_value, type, upper);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

template <typename T>
using ThresholdFixture = ThresholdValidationFixture<Tensor, Accessor, NEThreshold, T>;

FIXTURE_DATA_TEST_CASE(RunSmall, ThresholdFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), datasets::MixedThresholdDataset()),
                                                                                                       framework::dataset::make("DataType",
                                                                                                               DataType::U8)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, ThresholdFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), datasets::MixedThresholdDataset()),
                                                                                                     framework::dataset::make("DataType",
                                                                                                             DataType::U8)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
