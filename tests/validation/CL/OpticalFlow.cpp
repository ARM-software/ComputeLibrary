/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "arm_compute/runtime/CL/CLArray.h"
#include "arm_compute/runtime/CL/CLPyramid.h"
#include "arm_compute/runtime/CL/functions/CLGaussianPyramid.h"
#include "arm_compute/runtime/CL/functions/CLOpticalFlow.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/CL/CLAccessor.h"
#include "tests/CL/CLArrayAccessor.h"
#include "tests/datasets/BorderModeDataset.h"
#include "tests/datasets/OpticalFlowDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/OpticalFlowFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(OpticalFlow)

// *INDENT-OFF*
// clang-format off
using CLOpticalFlowFixture = OpticalFlowValidationFixture<CLTensor,
                                                          CLAccessor,
                                                          CLKeyPointArray,
                                                          CLArrayAccessor<KeyPoint>,
                                                          CLOpticalFlow,
                                                          CLPyramid,
                                                          CLGaussianPyramidHalf,
                                                          uint8_t>;

FIXTURE_DATA_TEST_CASE(RunSmall, CLOpticalFlowFixture, framework::DatasetMode::NIGHTLY, combine(combine(
                       datasets::SmallOpticalFlowDataset(),
                       framework::dataset::make("Format", Format::U8)),
                       datasets::BorderModes()))
{
    // Validate output
    CLArrayAccessor<KeyPoint> array(_target);

    validate_keypoints(array.buffer(),
                       array.buffer() + array.num_values(),
                       _reference.begin(),
                       _reference.end());
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLOpticalFlowFixture, framework::DatasetMode::NIGHTLY, combine(combine(
                       datasets::LargeOpticalFlowDataset(),
                       framework::dataset::make("Format", Format::U8)),
                       datasets::BorderModes()))
{
    // Validate output
    CLArrayAccessor<KeyPoint> array(_target);

    validate_keypoints(array.buffer(),
                       array.buffer() + array.num_values(),
                       _reference.begin(),
                       _reference.end());
}
// clang-format on
// *INDENT-ON*

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
