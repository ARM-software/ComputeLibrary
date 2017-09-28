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
#include "arm_compute/runtime/NEON/functions/NEDirectConvolutionLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/DirectConvolutionLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr AbsoluteTolerance<float> tolerance_qs(1.f); /**< Tolerance for fixed point tests */
#ifdef ARM_COMPUTE_ENABLE_FP16
constexpr AbsoluteTolerance<float> tolerance_fp16(0.01f);  /**< Tolerance for half precision floating point tests */
#endif                                                     /* ARM_COMPUTE_ENABLE_FP16 */
constexpr AbsoluteTolerance<float> tolerance_fp32(0.001f); /**< Tolerance for floating point tests */

/** Direct convolution data set. */
const auto data_pad_f32 = concat(concat(combine(framework::dataset::make("PadX", 0),
                                                combine(framework::dataset::make("PadY", 0),
                                                        framework::dataset::make("KernelSize", 1))),
                                        combine(framework::dataset::make("PadX", 0, 2),
                                                combine(framework::dataset::make("PadY", 0, 2),
                                                        framework::dataset::make("KernelSize", 3)))),
                                 combine(framework::dataset::make("PadX", 0, 3),
                                         combine(framework::dataset::make("PadY", 0, 3),
                                                 framework::dataset::make("KernelSize", 5))));

const auto data_pad_qs8 = concat(combine(framework::dataset::make("PadX", 0),
                                         combine(framework::dataset::make("PadY", 0),
                                                 framework::dataset::make("KernelSize", 1))),
                                 combine(framework::dataset::make("PadX", 0, 2),
                                         combine(framework::dataset::make("PadY", 0, 2),
                                                 framework::dataset::make("KernelSize", 3))));

const auto data_f32 = combine(datasets::SmallDirectConvolutionShapes(),
                              combine(framework::dataset::make("StrideX", 1, 3),
                                      combine(framework::dataset::make("StrideY", 1, 3),
                                              combine(data_pad_f32,
                                                      framework::dataset::make("NumKernels", { 1, 4, 8, 16 })))));

const auto data_qs8 = combine(datasets::SmallDirectConvolutionShapes(),
                              combine(framework::dataset::make("StrideX", 1, 3),
                                      combine(framework::dataset::make("StrideY", 1, 3),
                                              combine(data_pad_qs8,
                                                      framework::dataset::make("NumKernels", { 1, 4, 8, 16 })))));

/** Direct convolution QS16 data set. */
const auto data_qs16 = combine(datasets::SmallDirectConvolutionShapes(),
                               combine(framework::dataset::make("StrideX", 1, 3),
                                       combine(framework::dataset::make("StrideY", 1, 3),
                                               combine(framework::dataset::make("PadX", 0),
                                                       combine(framework::dataset::make("PadY", 0),
                                                               combine(framework::dataset::make("KernelSize", 1),
                                                                       framework::dataset::make("NumKernels", { 1, 4, 8, 16 })))))));
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(DirectConvolutionLayer)

template <typename T>
using NEDirectConvolutionLayerFixture = DirectConvolutionValidationFixture<Tensor, Accessor, NEDirectConvolutionLayer, T>;

TEST_SUITE(Float)
#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(Run, NEDirectConvolutionLayerFixture<half>, framework::DatasetMode::ALL, combine(data_f32, framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp16);
}
TEST_SUITE_END()
#endif /* ARM_COMPUTE_ENABLE_FP16 */

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(Run, NEDirectConvolutionLayerFixture<float>, framework::DatasetMode::ALL, combine(data_f32, framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}
TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using NEDirectConvolutionLayerFixedPointFixture = DirectConvolutionValidationFixedPointFixture<Tensor, Accessor, NEDirectConvolutionLayer, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QS8)
// We test for fixed point precision [4,6]
FIXTURE_DATA_TEST_CASE(Run, NEDirectConvolutionLayerFixedPointFixture<int8_t>, framework::DatasetMode::ALL, combine(combine(data_qs8, framework::dataset::make("DataType", DataType::QS8)),
                                                                                                                    framework::dataset::make("FractionalBits", 4, 7)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qs);
}
TEST_SUITE_END()

TEST_SUITE(QS16)
// We test for fixed point precision [4,13]
FIXTURE_DATA_TEST_CASE(Run, NEDirectConvolutionLayerFixedPointFixture<int16_t>, framework::DatasetMode::ALL, combine(combine(data_qs16, framework::dataset::make("DataType", DataType::QS16)),
                                                                                                                     framework::dataset::make("FractionalBits", 4, 14)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qs);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
