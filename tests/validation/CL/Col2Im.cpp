/*
 * Copyright (c) 2018-2020 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLCol2ImKernel.h"
#include "arm_compute/core/Types.h"

#include "tests/CL/CLAccessor.h"
#include "tests/CL/Helper.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/Col2ImFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(Col2Im)

using CLCol2Im = CLSynthetizeFunction<CLCol2ImKernel>;

/** Negative tests
 *
 * A series of validation tests on configurations which according to the API specification
 * the function should fail against.
 *
 * Checks performed in order:
 *     - Pass unsupported data type for input
 *     - Pass NHWC as output data layout
 *     - Pass an invalid output shape
 */
TEST_CASE(Negative, framework::DatasetMode::PRECOMMIT)
{
    // Unsupported data type
    {
        const auto input     = TensorInfo(TensorShape(10U, 12U, 1U, 2U), 1, DataType::SIZET);
        const auto output    = TensorInfo(TensorShape(3U, 4U, 10U, 1U, 2U), 1, DataType::F32);
        const auto conv_size = Size2D(3, 4);
        const auto status    = CLCol2ImKernel::validate(&input, &output, conv_size);
        ARM_COMPUTE_EXPECT(bool(status) == false, framework::LogLevel::ERRORS);
    }

    // NHWC as output data layout
    {
        const auto input     = TensorInfo(TensorShape(10U, 12U, 1U, 2U), 1, DataType::F32);
        const auto output    = TensorInfo(TensorShape(3U, 4U, 10U, 1U, 2U), 1, DataType::F32, DataLayout::NHWC);
        const auto conv_size = Size2D(3, 4);
        const auto status    = CLCol2ImKernel::validate(&input, &output, conv_size);
        ARM_COMPUTE_EXPECT(bool(status) == false, framework::LogLevel::ERRORS);
    }

    // Invalid output size
    {
        const auto input     = TensorInfo(TensorShape(10U, 12U, 1U, 2U), 1, DataType::F32);
        const auto output    = TensorInfo(TensorShape(3U, 4U, 10U, 2U, 2U), 1, DataType::F32);
        const auto conv_size = Size2D(3, 4);
        const auto status    = CLCol2ImKernel::validate(&input, &output, conv_size);
        ARM_COMPUTE_EXPECT(bool(status) == false, framework::LogLevel::ERRORS);
    }
}

template <typename T>
using CLCol2ImFixture = Col2ImValidationFixture<CLTensor, CLAccessor, CLCol2Im, T, true>;

/** Test kernel for single-precision floating point
 *
 * @note 8 elements processed per iteration
 *
 * Three main tests will be run:
 *  - Channels are multiple of elements processed
 *  - Channels larger and non multiple of elements used
 *  - Channels smaller and not multiple of elements used
 *
 *  The above will be repeated with a different group size
 *
 *  Kernel tested col2im
 */
FIXTURE_DATA_TEST_CASE(FP32,
                       CLCol2ImFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(
                                                   framework::dataset::make("InputShape", { TensorShape(8U, 16U, 3U, 1U), TensorShape(17U, 16U, 3U, 1U), TensorShape(7U, 16U, 3U, 1U) }),
                                                   framework::dataset::make("ConvolvedWidth", 4)),
                                               framework::dataset::make("ConvolvedHeight", 4)),
                                       framework::dataset::make("Groups", { 1, 3 })),
                               framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

/** Test kernel for half-precision floating point
 *
 * @note 8 elements processed per iteration
 *
 * One main tests will be run:
 *  - Channels larger and non multiple of elements used
 *
 *  We just need to test the difference in the data type size.
 *  Any other issues can be identified by the main FP32 tests
 *
 *  Kernel tested col2im
 */
FIXTURE_DATA_TEST_CASE(F16,
                       CLCol2ImFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(
                                                   framework::dataset::make("InputShape", TensorShape(17U, 16U, 3U, 1U)),
                                                   framework::dataset::make("ConvolvedWidth", 4)),
                                               framework::dataset::make("ConvolvedHeight", 4)),
                                       framework::dataset::make("Groups", 3)),
                               framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

/** Test kernel for unsigned asymmetric quantized type
 *
 * @note 8 elements processed per iteration
 *
 * One main tests will be run:
 *  - Channels larger and non multiple of elements used
 *
 *  We just need to test the difference in the data type size.
 *  Any other issues can be identified by the main FP32 tests
 *
 *  Kernel tested col2im
 */
FIXTURE_DATA_TEST_CASE(QASYMM8,
                       CLCol2ImFixture<uint8_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(
                                                   framework::dataset::make("InputShape", TensorShape(17U, 16U, 3U, 1U)),
                                                   framework::dataset::make("ConvolvedWidth", 4)),
                                               framework::dataset::make("ConvolvedHeight", 4)),
                                       framework::dataset::make("Groups", 3)),
                               framework::dataset::make("DataType", DataType::QASYMM8)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

TEST_SUITE_END() // CL
TEST_SUITE_END() // Col2Im
} // namespace validation
} // namespace test
} // namespace arm_compute
