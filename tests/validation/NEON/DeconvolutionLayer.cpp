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
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEDeconvolutionLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/DeconvolutionLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr AbsoluteTolerance<float> tolerance_fp32(0.001f); /**< Tolerance for floating point tests */

const auto data3x3 = datasets::SmallDeconvolutionShapes() * framework::dataset::make("StrideX", 1, 4) * framework::dataset::make("StrideY", 1, 4) * framework::dataset::make("PadX", 0, 2)
                     * framework::dataset::make("PadY", 0, 2) * framework::dataset::make("ax", 0) * framework::dataset::make("ay", 0) * framework::dataset::make("NumKernels", { 1, 3 });

const auto data1x1 = datasets::SmallDeconvolutionShapes() * framework::dataset::make("StrideX", 1, 4) * framework::dataset::make("StrideY", 1, 4) * framework::dataset::make("PadX", 0, 1)
                     * framework::dataset::make("PadY", 0, 1) * framework::dataset::make("ax", 0) * framework::dataset::make("ay", 0) * framework::dataset::make("NumKernels", { 1, 3 });

} // namespace

TEST_SUITE(NEON)
TEST_SUITE(DeconvolutionLayer)

template <typename T>
using NEDeconvolutionLayerFixture3x3 = DeconvolutionValidationFixture<Tensor, Accessor, NEDeconvolutionLayer, T, 3, 3>;

template <typename T>
using NEDeconvolutionLayerFixture1x1 = DeconvolutionValidationFixture<Tensor, Accessor, NEDeconvolutionLayer, T, 1, 1>;

TEST_SUITE(Float)

TEST_SUITE(FP32)
TEST_SUITE(W3x3)

FIXTURE_DATA_TEST_CASE(Run, NEDeconvolutionLayerFixture3x3<float>, framework::DatasetMode::ALL, combine(data3x3, framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}
TEST_SUITE_END()

TEST_SUITE(W1x1)
FIXTURE_DATA_TEST_CASE(Run, NEDeconvolutionLayerFixture1x1<float>, framework::DatasetMode::ALL, combine(data1x1, framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
