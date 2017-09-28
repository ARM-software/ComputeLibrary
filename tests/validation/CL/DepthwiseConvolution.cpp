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
 * OUT OF OR IN CONCLCTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLDepthwiseConvolution.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/DepthwiseConvolutionDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/DepthwiseConvolutionFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr RelativeTolerance<float> tolerance_f32(0.01f); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
} // namespace

TEST_SUITE(CL)
TEST_SUITE(DepthwiseConvolutionLayer)

template <typename T>
using CLDepthwiseConvolutionFixture = DepthwiseConvolutionValidationFixture<CLTensor, CLAccessor, CLDepthwiseConvolution, T>;

TEST_SUITE(Generic)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthwiseConvolutionFixture<float>, framework::DatasetMode::PRECOMMIT, datasets::SmallDepthwiseConvolutionDataset())
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthwiseConvolutionFixture<float>, framework::DatasetMode::NIGHTLY, datasets::LargeDepthwiseConvolutionDataset())
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END()

template <typename T>
using CLDepthwiseConvolutionFixture3x3 = DepthwiseConvolutionValidationFixture<CLTensor, CLAccessor, CLDepthwiseConvolution3x3, T>;

TEST_SUITE(W3x3)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthwiseConvolutionFixture3x3<float>, framework::DatasetMode::PRECOMMIT, datasets::SmallDepthwiseConvolutionDataset3x3())
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthwiseConvolutionFixture3x3<float>, framework::DatasetMode::NIGHTLY, datasets::LargeDepthwiseConvolutionDataset3x3())
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
