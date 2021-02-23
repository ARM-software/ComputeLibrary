/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEDilate.h"
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
#include "tests/validation/fixtures/DilateFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr unsigned int filter_size = 3;              /* Size of the kernel/filter in number of elements. */
constexpr BorderSize   border_size(filter_size / 2); /* Border size of the kernel/filter around its central element. */
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(Dilate)

template <typename T>
using NEDilateFixture = DilateValidationFixture<Tensor, Accessor, NEDilate, T>;

FIXTURE_DATA_TEST_CASE(RunSmall, NEDilateFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType",
                                                                                                            DataType::U8)),
                                                                                                    datasets::BorderModes()))
{
    // Validate output
    validate(Accessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), border_size));
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDilateFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType",
                                                                                                            DataType::U8)),
                                                                                                    datasets::BorderModes()))
{
    // Validate output
    validate(Accessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), border_size));
}

TEST_SUITE_END() // Dilate
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
