/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLFastCorners.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/CL/CLAccessor.h"
#include "tests/CL/CLArrayAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ImageFileDatasets.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/FastCornersFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/* Tolerance used to compare corner strengths */
const AbsoluteTolerance<float> tolerance(0.5f);
} // namespace

TEST_SUITE(CL)
TEST_SUITE(FastCorners)

template <typename T>
using CLFastCornersFixture = FastCornersValidationFixture<CLTensor, CLAccessor, CLKeyPointArray, CLFastCorners, T>;

FIXTURE_DATA_TEST_CASE(RunSmall, CLFastCornersFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallImageFiles(), framework::dataset::make("Format", Format::U8)),
                                                                                                                   framework::dataset::make("SuppressNonMax", { false, true })),
                                                                                                           framework::dataset::make("BorderMode", BorderMode::UNDEFINED)))
{
    // Validate output
    CLArrayAccessor<KeyPoint> array(_target);
    validate_keypoints(array.buffer(), array.buffer() + array.num_values(), _reference.begin(), _reference.end(), tolerance);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLFastCornersFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeImageFiles(), framework::dataset::make("Format", Format::U8)),
                                                                                                                 framework::dataset::make("SuppressNonMax", { false, true })),
                                                                                                         framework::dataset::make("BorderMode", BorderMode::UNDEFINED)))
{
    // Validate output
    CLArrayAccessor<KeyPoint> array(_target);
    validate_keypoints(array.buffer(), array.buffer() + array.num_values(), _reference.begin(), _reference.end(), tolerance);
}

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
