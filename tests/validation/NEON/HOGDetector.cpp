/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "arm_compute/runtime/Array.h"
#include "arm_compute/runtime/NEON/functions/NEHOGDescriptor.h"
#include "arm_compute/runtime/NEON/functions/NEHOGDetector.h"
#include "tests/NEON/Accessor.h"
#include "tests/NEON/ArrayAccessor.h"
#include "tests/NEON/HOGAccessor.h"
#include "tests/datasets/HOGDescriptorDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/HOGDetectorFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/* Set the tolerance (percentage) used when validating the score of detection window.
   Note: High tolerance is required due to divergence between CL and Neon detection window scores. */
RelativeTolerance<float> tolerance(1.0f);

/* Input dataset (values must be a multiple of the HOGInfo block_size) */
const auto DetectionWindowStrideDataset = framework::dataset::make("DetectionWindowStride", { Size2D(8, 8), Size2D(16, 16) });
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(HOGDetector)

// *INDENT-OFF*
// clang-format off
using NEHOGDetectorFixture = HOGDetectorValidationFixture<Tensor,
                                                          HOG,
                                                          DetectionWindowArray,
                                                          NEHOGDescriptor,
                                                          Accessor,
                                                          ArrayAccessor<DetectionWindow>,
                                                          HOGAccessor,
                                                          NEHOGDetector,
                                                          uint8_t,
                                                          float>;

FIXTURE_DATA_TEST_CASE(RunSmall, NEHOGDetectorFixture, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(
                       DetectionWindowStrideDataset,
                       datasets::SmallHOGDescriptorDataset()),
                       framework::dataset::make("Format", Format::U8)),
                       framework::dataset::make("BorderMode", {BorderMode::CONSTANT, BorderMode::REPLICATE})))
{
    // Validate output
    validate_detection_windows(_target.begin(), _target.end(), _reference.begin(), _reference.end(), tolerance);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEHOGDetectorFixture, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(
                       DetectionWindowStrideDataset,
                       datasets::LargeHOGDescriptorDataset()),
                       framework::dataset::make("Format", Format::U8)),
                       framework::dataset::make("BorderMode", {BorderMode::CONSTANT, BorderMode::REPLICATE})))
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
