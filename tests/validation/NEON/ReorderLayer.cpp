/*
 * Copyright (c) 2023 Arm Limited.
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
#if defined(__aarch64__)

#include "arm_compute/runtime/NEON/functions/NEReorderLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/NEON/Accessor.h"
#include "tests/datasets/ReorderLayerDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ReorderFixture.h"
#include "src/core/NEON/kernels/NEReorderKernel.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(NEON)
TEST_SUITE(ReorderLayer)

template <typename T>
using NEReorderLayerAlias = ReorderValidationFixture<Tensor, Accessor, NEReorderLayer, T>;

TEST_SUITE(FP32)
#if defined(ARM_COMPUTE_ENABLE_SVE)
FIXTURE_DATA_TEST_CASE(RunBlock8, NEReorderLayerAlias<float>, framework::DatasetMode::ALL, combine(datasets::ReorderLayerDatasetBlock8(), framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
#endif // ARM_COMPUTE_ENABLE_SVE
FIXTURE_DATA_TEST_CASE(RunBlock4, NEReorderLayerAlias<float>, framework::DatasetMode::ALL, combine(datasets::ReorderLayerDatasetBlock4(), framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

TEST_SUITE_END() // FP32

TEST_SUITE_END() // ReorderLayer
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute

#endif  // defined(__aarch64__)