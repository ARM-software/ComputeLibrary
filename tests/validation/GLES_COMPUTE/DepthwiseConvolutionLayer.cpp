/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/runtime/GLES_COMPUTE/GCTensor.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCTensorAllocator.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCDepthwiseConvolutionLayer.h"
#include "tests/GLES_COMPUTE/GCAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/DepthwiseConvolutionLayerDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/DepthwiseConvolutionLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
RelativeTolerance<half> tolerance_fp16(half(0.2)); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F16 */
constexpr float         tolerance_num = 0.07f;     /**< Tolerance number */

const auto depth_multipliers = framework::dataset::make("DepthMultiplier", { 1, 2, 3 });

//Activation Functions
const auto ActivationFunctionsEmptyDataset = framework::dataset::make("ActivationInfo",
{ ActivationLayerInfo() });
} // namespace

TEST_SUITE(GC)
TEST_SUITE(DepthwiseConvolutionLayer)

template <typename T>
using GCDepthwiseConvolutionLayerFixture3x3 = DepthwiseConvolutionLayerValidationFixture<GCTensor, GCAccessor, GCDepthwiseConvolutionLayer3x3, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
TEST_SUITE(W3x3)
FIXTURE_DATA_TEST_CASE(RunSmall, GCDepthwiseConvolutionLayerFixture3x3<half>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset3x3(),
                                                                                                                   depth_multipliers),
                                                                                                                   framework::dataset::make("DataType",
                                                                                                                           DataType::F16)),
                                                                                                                   framework::dataset::make("DataLayout", DataLayout::NCHW)),
                                                                                                                   ActivationFunctionsEmptyDataset))
{
    validate(GCAccessor(_target), _reference, tolerance_fp16, tolerance_num);
}
FIXTURE_DATA_TEST_CASE(RunLarge, GCDepthwiseConvolutionLayerFixture3x3<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeDepthwiseConvolutionLayerDataset3x3(),
                                                                                                                       depth_multipliers),
                                                                                                                       framework::dataset::make("DataType",
                                                                                                                               DataType::F16)),
                                                                                                                       framework::dataset::make("DataLayout", DataLayout::NCHW)),
                                                                                                                       ActivationFunctionsEmptyDataset))
{
    validate(GCAccessor(_target), _reference, tolerance_fp16, tolerance_num);
}
TEST_SUITE_END()
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
