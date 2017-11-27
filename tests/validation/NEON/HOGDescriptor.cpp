/*
 * Copyright (c) 2017, 2018 ARM Limited.
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
#include "arm_compute/runtime/HOG.h"
#include "arm_compute/runtime/NEON/functions/NEHOGDescriptor.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/BorderModeDataset.h"
#include "tests/datasets/HOGDescriptorDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/HOGDescriptorFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
AbsoluteTolerance<float> tolerance(0.5f);
constexpr float          tolerance_number = 0.01f;
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(HOGDescriptor)

// *INDENT-OFF*
// clang-format off
using NEHOGDescriptorFixture = HOGDescriptorValidationFixture<Tensor, HOG, Accessor, NEHOGDescriptor, uint8_t, float>;

FIXTURE_DATA_TEST_CASE(RunSmall, NEHOGDescriptorFixture, framework::DatasetMode::PRECOMMIT,
                       combine(combine(
                       datasets::SmallHOGDescriptorDataset(),
                       framework::dataset::make("Format", Format::U8)),
                       framework::dataset::make("BorderMode", {BorderMode::CONSTANT, BorderMode::REPLICATE})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance, tolerance_number);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEHOGDescriptorFixture, framework::DatasetMode::NIGHTLY,
                       combine(combine(
                       datasets::LargeHOGDescriptorDataset(),
                       framework::dataset::make("Format", Format::U8)),
                       framework::dataset::make("BorderMode", {BorderMode::CONSTANT, BorderMode::REPLICATE})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance, tolerance_number);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
