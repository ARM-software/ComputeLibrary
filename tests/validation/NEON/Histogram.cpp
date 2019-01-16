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
#include "arm_compute/runtime/Distribution1D.h"
#include "arm_compute/runtime/NEON/functions/NEHistogram.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/BorderModeDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/HistogramFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(NEON)
TEST_SUITE(Histogram)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(datasets::Small2DShapes(),
                                                                   framework::dataset::make("DataType", DataType::U8)),
               shape, data_type)
{
    // Setup Distribution
    std::mt19937                            gen(library->seed());
    std::uniform_int_distribution<size_t>   distribution_size_t(1, 30);
    const size_t                            num_bins = distribution_size_t(gen);
    std::uniform_int_distribution<int32_t>  distribution_int32_t(0, 125);
    const size_t                            offset = distribution_int32_t(gen);
    std::uniform_int_distribution<uint32_t> distribution_uint32_t(1, 255 - offset);
    const size_t                            range = distribution_uint32_t(gen);
    Distribution1D                          distribution_dst(num_bins, offset, range);

    // Create tensors
    Tensor      src = create_tensor<Tensor>(shape, data_type);
    TensorShape dst_shape(num_bins);
    Tensor      dst = create_tensor<Tensor>(dst_shape, DataType::U32);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    NEHistogram histogram;
    histogram.configure(&src, &distribution_dst);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(src.info()->valid_region(), valid_region);
    const ValidRegion valid_region_dst = shape_to_valid_region(dst_shape);
    validate(dst.info()->valid_region(), valid_region_dst);

    // Validate padding
    const PaddingSize padding;
    validate(src.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

template <typename T>
using NEHistogramFixture = HistogramValidationFixture<Tensor, Accessor, NEHistogram, T, Distribution1D>;

FIXTURE_DATA_TEST_CASE(RunSmall, NEHistogramFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(datasets::Small2DShapes(), framework::dataset::make("DataType",
                                                                                                         DataType::U8)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEHistogramFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::Large2DShapes(), framework::dataset::make("DataType",
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
