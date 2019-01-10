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
#include "arm_compute/runtime/CL/functions/CLMagnitude.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/MagnitudeFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
template <typename T>
AbsoluteTolerance<T> tolerance(MagnitudeType magnitude_type)
{
    return AbsoluteTolerance<T>((MagnitudeType::L1NORM == magnitude_type) ? 0 : 1);
}
} // namespace

TEST_SUITE(CL)
TEST_SUITE(Magnitude)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), framework::dataset::make("DataType", { DataType::S16, DataType::S32 })),
               shape, data_type)
{
    // Create tensors
    CLTensor src1 = create_tensor<CLTensor>(shape, data_type);
    CLTensor src2 = create_tensor<CLTensor>(shape, data_type);
    CLTensor dst  = create_tensor<CLTensor>(shape, data_type);

    ARM_COMPUTE_EXPECT(src1.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(src2.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function (default MagnitudeType::L2NORM)
    CLMagnitude magnitude;
    magnitude.configure(&src1, &src2, &dst);

    // Validate valid region
    const ValidRegion valid_region = shape_to_valid_region(shape);
    validate(dst.info()->valid_region(), valid_region);

    // Validate padding
    const PaddingSize padding = PaddingCalculator(shape.x(), 16).required_padding();

    validate(src1.info()->padding(), padding);
    validate(src2.info()->padding(), padding);
    validate(dst.info()->padding(), padding);
}

template <typename T>
using CLMagnitudeFixture = MagnitudeValidationFixture<CLTensor, CLAccessor, CLMagnitude, T>;

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLMagnitudeFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::Small2DShapes(), framework::dataset::make("Format", Format::S16)),
                                                                                                         framework::dataset::make("MagnitudeType", { MagnitudeType::L1NORM, MagnitudeType::L2NORM })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance<int16_t>(_magnitude_type));
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLMagnitudeFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::Large2DShapes(), framework::dataset::make("Format", Format::S16)),
                                                                                                       framework::dataset::make("MagnitudeType", { MagnitudeType::L1NORM, MagnitudeType::L2NORM })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance<int16_t>(_magnitude_type));
}
TEST_SUITE_END() // S16

TEST_SUITE(S32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLMagnitudeFixture<int32_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::Small2DShapes(), framework::dataset::make("Format", Format::S32)),
                                                                                                         framework::dataset::make("MagnitudeType", { MagnitudeType::L1NORM, MagnitudeType::L2NORM })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance<int32_t>(_magnitude_type));
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLMagnitudeFixture<int32_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::Large2DShapes(), framework::dataset::make("Format", Format::S32)),
                                                                                                       framework::dataset::make("MagnitudeType", { MagnitudeType::L1NORM, MagnitudeType::L2NORM })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance<int32_t>(_magnitude_type));
}
TEST_SUITE_END() // S32

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
