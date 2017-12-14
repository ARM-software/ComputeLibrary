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
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLAccumulate.h"
#include "tests/CL/CLAccessor.h"
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
TEST_SUITE(CL)
TEST_SUITE(Accumulate)

TEST_SUITE(U8)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(framework::dataset::concat(datasets::SmallShapes(), datasets::LargeShapes()), AccumulateS16Dataset),
               shape, data_type, output_data_type)
{
    // Create tensors
    CLTensor ref_src = create_tensor<CLTensor>(shape, data_type);
    CLTensor dst     = create_tensor<CLTensor>(shape, output_data_type);

    // Create and Configure function
    CLAccumulate accum;
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
using CLAccumulateFixture = AccumulateValidationFixture<CLTensor, CLAccessor, CLAccumulate, T1, int16_t>;

FIXTURE_DATA_TEST_CASE(RunSmall, CLAccumulateFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallShapes(), AccumulateS16Dataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLAccumulateFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), AccumulateS16Dataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance);
}

TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE(AccumulateWeighted)

TEST_SUITE(U8)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(framework::dataset::concat(datasets::SmallShapes(), datasets::LargeShapes()), AccumulateU8Dataset),
               shape, data_type, output_data_type)
{
    // Generate a random alpha value
    std::mt19937                     gen(library->seed());
    std::uniform_real_distribution<> float_dist(0, 1);
    const float                      alpha = float_dist(gen);

    // Create tensors
    CLTensor ref_src = create_tensor<CLTensor>(shape, data_type);
    CLTensor dst     = create_tensor<CLTensor>(shape, output_data_type);

    // Create and Configure function
    CLAccumulateWeighted accum_weight;
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
using CLAccumulateWeightedFixture = AccumulateWeightedValidationFixture<CLTensor, CLAccessor, CLAccumulateWeighted, T1, uint8_t>;

FIXTURE_DATA_TEST_CASE(RunSmall, CLAccumulateWeightedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallShapes(), AccumulateU8Dataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLAccumulateWeightedFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), AccumulateU8Dataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance);
}

TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE(AccumulateSquared)

TEST_SUITE(U8)
DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(framework::dataset::concat(datasets::SmallShapes(), datasets::LargeShapes()), AccumulateS16Dataset),
               shape, data_type, output_data_type)
{
    // Generate a random shift value
    std::mt19937                            gen(library->seed());
    std::uniform_int_distribution<uint32_t> int_dist(0, 15);
    const uint32_t                          shift = int_dist(gen);

    // Create tensors
    CLTensor ref_src = create_tensor<CLTensor>(shape, data_type);
    CLTensor dst     = create_tensor<CLTensor>(shape, output_data_type);

    // Create and Configure function
    CLAccumulateSquared accum_square;
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
using CLAccumulateSquaredFixture = AccumulateSquaredValidationFixture<CLTensor, CLAccessor, CLAccumulateSquared, T1, int16_t>;

FIXTURE_DATA_TEST_CASE(RunSmall, CLAccumulateSquaredFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallShapes(), AccumulateS16Dataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLAccumulateSquaredFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), AccumulateS16Dataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance);
}

TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
