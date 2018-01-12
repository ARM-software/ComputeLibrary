/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLDepthwiseConvolutionLayer.h"
#include "tests/CL/CLAccessor.h"
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
constexpr RelativeTolerance<float>   tolerance_f32(0.01f); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(1); /**< Tolerance value for comparing reference's output against implementation's output for DataType::QASYMM8 */
} // namespace

TEST_SUITE(CL)
TEST_SUITE(DepthwiseConvolutionLayer)

template <typename T>
using CLDepthwiseConvolutionLayerFixture = DepthwiseConvolutionLayerValidationFixture<CLTensor, CLAccessor, CLDepthwiseConvolutionLayer, T>;

TEST_SUITE(Generic)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::ALL, combine(datasets::SmallDepthwiseConvolutionLayerDataset(), framework::dataset::make("DataType",
                                                                                                                 DataType::F32)))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthwiseConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeDepthwiseConvolutionLayerDataset(),
                                                                                                                     framework::dataset::make("DataType",
                                                                                                                             DataType::F32)))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END()

template <typename T>
using CLDepthwiseConvolutionLayerFixture3x3 = DepthwiseConvolutionLayerValidationFixture<CLTensor, CLAccessor, CLDepthwiseConvolutionLayer3x3, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
TEST_SUITE(W3x3)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthwiseConvolutionLayerFixture3x3<float>, framework::DatasetMode::ALL, combine(datasets::SmallDepthwiseConvolutionLayerDataset3x3(),
                                                                                                                    framework::dataset::make("DataType",
                                                                                                                            DataType::F32)))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthwiseConvolutionLayerFixture3x3<float>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeDepthwiseConvolutionLayerDataset3x3(),
                                                                                                                        framework::dataset::make("DataType",
                                                                                                                                DataType::F32)))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END()
TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using CLDepthwiseConvolutionLayerQuantizedFixture3x3 = DepthwiseConvolutionLayerValidationQuantizedFixture<CLTensor, CLAccessor, CLDepthwiseConvolutionLayer3x3, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
TEST_SUITE(W3x3)
FIXTURE_DATA_TEST_CASE(RunSmall, CLDepthwiseConvolutionLayerQuantizedFixture3x3<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset3x3(),
                       framework::dataset::make("DataType", DataType::QASYMM8)),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, 10) })))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLDepthwiseConvolutionLayerQuantizedFixture3x3<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeDepthwiseConvolutionLayerDataset3x3(),
                       framework::dataset::make("DataType", DataType::QASYMM8)),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.5f, 10) })))
{
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END()
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
