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
#include "arm_compute/runtime/NEON/functions/NEReductionOperation.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ReductionOperationDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ReductionOperationFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Tolerance for float operations */
RelativeTolerance<float> tolerance_f32(0.00001f);
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(ReductionOperation)

template <typename T>
using NEReductionOperationFixture = ReductionOperationValidationFixture<Tensor, Accessor, NEReductionOperation, T>;

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEReductionOperationFixture<float>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType", DataType::F32)), framework::dataset::make("Axis", { 0 })), datasets::ReductionOperations()))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEReductionOperationFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType", DataType::F32)), framework::dataset::make("Axis", { 0 })), datasets::ReductionOperations()))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
