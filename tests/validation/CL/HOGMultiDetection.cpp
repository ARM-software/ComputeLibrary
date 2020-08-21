/*
 * Copyright (c) 2018 Arm Limited.
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
#include "arm_compute/runtime/CL/CLMultiHOG.h"
#include "arm_compute/runtime/CL/functions/CLHOGDescriptor.h"
#include "arm_compute/runtime/CL/functions/CLHOGMultiDetection.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/CL/CLAccessor.h"
#include "tests/CL/CLArrayAccessor.h"
#include "tests/CL/CLHOGAccessor.h"
#include "tests/datasets/HOGMultiDetectionDataset.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/HOGMultiDetectionFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/* Set the tolerance (percentage) used when validating the strength of detection window. */
RelativeTolerance<float> tolerance(0.1f);
} // namespace

TEST_SUITE(CL)
TEST_SUITE(HOGMultiDetection)

// *INDENT-OFF*
// clang-format off
using CLHOGMultiDetectionFixture = HOGMultiDetectionValidationFixture<CLTensor,
                                                                      CLHOG,
                                                                      CLMultiHOG,
                                                                      CLDetectionWindowArray,
                                                                      CLSize2DArray,
                                                                      CLAccessor,
                                                                      CLArrayAccessor<Size2D>,
                                                                      CLArrayAccessor<DetectionWindow>,
                                                                      CLHOGAccessor,
                                                                      CLHOGMultiDetection,
                                                                      uint8_t,
                                                                      float>;


FIXTURE_DATA_TEST_CASE(RunSmall, CLHOGMultiDetectionFixture, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(
                       datasets::SmallHOGMultiDetectionDataset(),
                       framework::dataset::make("Format", Format::U8)),
                       framework::dataset::make("BorderMode", {BorderMode::CONSTANT, BorderMode::REPLICATE})),
                       framework::dataset::make("NonMaximaSuppression", {false, true})))
{
    // Validate output
    validate_detection_windows(_target.begin(), _target.end(), _reference.begin(), _reference.end(), tolerance);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLHOGMultiDetectionFixture, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(
                       datasets::LargeHOGMultiDetectionDataset(),
                       framework::dataset::make("Format", Format::U8)),
                       framework::dataset::make("BorderMode", {BorderMode::CONSTANT, BorderMode::REPLICATE})),
                       framework::dataset::make("NonMaximaSuppression", {false, true})))
{
    // Validate output
    validate_detection_windows(_target.begin(), _target.end(), _reference.begin(), _reference.end(), tolerance);
}

// clang-format on
// *INDENT-ON*

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
