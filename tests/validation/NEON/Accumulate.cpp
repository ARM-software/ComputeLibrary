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
#include "arm_compute/runtime/NEON/functions/NEAccumulate.h"
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
#include "tests/validation/fixtures/AccumulateFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Tolerance value for comparing reference's output against implementation's output for floating point data types */
constexpr AbsoluteTolerance<float> tolerance(1.0f);
/** Input data sets **/
const auto AccumulateU8Dataset  = combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::U8));
const auto AccumulateS16Dataset = combine(framework::dataset::make("DataType", DataType::U8), framework::dataset::make("DataType", DataType::S16));
} // namespace
TEST_SUITE(NEON)
TEST_SUITE(Accumulate)

TEST_SUITE(U8)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), AccumulateS16Dataset),
               shape, data_type, output_data_type)
{
    // Create tensors
    Tensor ref_src = create_tensor<Tensor>(shape, data_type);
    Tensor dst     = create_tensor<Tensor>(shape, output_data_type);

    // Create and Configure function
    NEAccumulate accum;
    accum.configure(&ref_src, &dst);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(ref_src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

template <typename T1>
using NEAccumulateFixture = AccumulateValidationFixture<Tensor, Accessor, NEAccumulate, T1, int16_t>;

FIXTURE_DATA_TEST_CASE(RunSmall, NEAccumulateFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallShapes(), AccumulateS16Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEAccumulateFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), AccumulateS16Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance);
}

TEST_SUITE_END() // U8
TEST_SUITE_END() // Accumulate

TEST_SUITE(AccumulateWeighted)

TEST_SUITE(U8)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), AccumulateU8Dataset),
               shape, data_type, output_data_type)
{
    // Generate a random alpha value
    std::mt19937                     gen(library->seed());
    std::uniform_real_distribution<> float_dist(0, 1);
    const float                      alpha = float_dist(gen);

    // Create tensors
    Tensor ref_src = create_tensor<Tensor>(shape, data_type);
    Tensor dst     = create_tensor<Tensor>(shape, output_data_type);

    // Create and Configure function
    NEAccumulateWeighted accum_weight;
    accum_weight.configure(&ref_src, alpha, &dst);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(ref_src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

template <typename T1>
using NEAccumulateWeightedFixture = AccumulateWeightedValidationFixture<Tensor, Accessor, NEAccumulateWeighted, T1, uint8_t>;

FIXTURE_DATA_TEST_CASE(RunSmall, NEAccumulateWeightedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallShapes(), AccumulateU8Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEAccumulateWeightedFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), AccumulateU8Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance);
}

TEST_SUITE_END() // U8
TEST_SUITE_END() // AccumulateWeighted

TEST_SUITE(AccumulateSquared)

TEST_SUITE(U8)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), AccumulateS16Dataset),
               shape, data_type, output_data_type)
{
    // Generate a random shift value
    std::mt19937                            gen(library->seed());
    std::uniform_int_distribution<uint32_t> int_dist(0, 15);
    const uint32_t                          shift = int_dist(gen);

    // Create tensors
    Tensor ref_src = create_tensor<Tensor>(shape, data_type);
    Tensor dst     = create_tensor<Tensor>(shape, output_data_type);

    // Create and Configure function
    NEAccumulateSquared accum_square;
    accum_square.configure(&ref_src, shift, &dst);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();
    validate(ref_src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

template <typename T1>
using NEAccumulateSquaredFixture = AccumulateSquaredValidationFixture<Tensor, Accessor, NEAccumulateSquared, T1, int16_t>;

FIXTURE_DATA_TEST_CASE(RunSmall, NEAccumulateSquaredFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallShapes(), AccumulateS16Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEAccumulateSquaredFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), AccumulateS16Dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance);
}

TEST_SUITE_END() // U8
TEST_SUITE_END() // AccumulateSquared

TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
