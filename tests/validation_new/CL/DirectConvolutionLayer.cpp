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
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLDirectConvolutionLayer.h"
#include "framework/Asserts.h"
#include "framework/Macros.h"
#include "framework/datasets/Datasets.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets_new/ShapeDatasets.h"
#include "tests/validation_new/Validation.h"
#include "tests/validation_new/fixtures/DirectConvolutionLayerFixture.h"
#include "tests/validation_new/half.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr float tolerance_fp16 = 0.1f;   /**< Tolerance for floating point tests */
constexpr float tolerance_fp32 = 0.001f; /**< Tolerance for floating point tests */

/** Direct convolution data set. */
const auto data = combine(datasets::SmallDirectConvolutionShapes(),
                          combine(framework::dataset::make("StrideX", 1, 3),
                                  combine(framework::dataset::make("StrideY", 1, 3),
                                          combine(concat(combine(framework::dataset::make("PadX", 0),
                                                                 combine(framework::dataset::make("PadY", 0),
                                                                         framework::dataset::make("KernelSize", 1))),
                                                         combine(framework::dataset::make("PadX", 0, 2),
                                                                 combine(framework::dataset::make("PadY", 0, 2),
                                                                         framework::dataset::make("KernelSize", 3)))),
                                                  framework::dataset::make("NumKernels", { 1, 4, 8, 16 })))));
} // namespace

TEST_SUITE(CL)
TEST_SUITE(DirectConvolutionLayer)

//TODO(COMPMID-415): Configuration tests?

template <typename T>
using CLDirectConvolutionLayerFixture = DirectConvolutionValidationFixture<CLTensor, CLAccessor, CLDirectConvolutionLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(Run, CLDirectConvolutionLayerFixture<half_float::half>, framework::DatasetMode::ALL, combine(data, framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp16);
}
TEST_SUITE_END()

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(Run, CLDirectConvolutionLayerFixture<float>, framework::DatasetMode::ALL, combine(data, framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp32);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
