/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "src/gpu/cl/kernels/ClIm2ColKernel.h"
#include "tests/CL/CLAccessor.h"
#include "tests/CL/Helper.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/Im2ColFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(Im2Col)

using ClIm2Col = ClSynthetizeOperatorWithBorder<opencl::kernels::ClIm2ColKernel>;

/** Negative tests
 *
 * A series of validation tests on configurations which according to the API specification
 * the function should fail against.
 *
 * Checks performed in order:
 *     - Pass unsupported data type for input
 *     - Pass a quantized input and ask to compress the bias into the resulting matrix
 *     - Pass a dilation factor of 0
 *     - Check NHWC data layout while requesting to perform a grouped operation
 *     - Check NCHW grouped operation when the number of channels is not multiple of the groups
 *     - Pass an invalid output shape
 */
TEST_CASE(Negative, framework::DatasetMode::ALL)
{
    // Unsupported data type
    {
        const auto input     = TensorInfo(TensorShape(10U, 12U, 1U, 2U), 1, DataType::SIZET);
        const auto output    = TensorInfo(TensorShape(9U, 10U, 12U, 2U), 1, DataType::F32);
        const auto conv_size = Size2D(3, 3);
        const bool has_bias  = false;
        const auto status    = opencl::kernels::ClIm2ColKernel::validate(&input, &output, conv_size, PadStrideInfo(), has_bias);
        ARM_COMPUTE_EXPECT(bool(status) == false, framework::LogLevel::ERRORS);
    }

    // Passing quantized input and ask to merge the bias in the output
    {
        const auto input     = TensorInfo(TensorShape(10U, 12U, 1U, 2U), 1, DataType::QASYMM8);
        const auto output    = TensorInfo(TensorShape(9U, 80U, 2U), 1, DataType::QASYMM8);
        const auto conv_size = Size2D(3, 3);
        const bool has_bias  = true;
        const auto status    = opencl::kernels::ClIm2ColKernel::validate(&input, &output, conv_size, PadStrideInfo(), has_bias);
        ARM_COMPUTE_EXPECT(bool(status) == false, framework::LogLevel::ERRORS);
    }

    // Invalid dilation
    {
        const auto input     = TensorInfo(TensorShape(10U, 12U, 1U, 2U), 1, DataType::F32);
        const auto output    = TensorInfo(TensorShape(9U, 80U, 2U), 1, DataType::F32);
        const auto conv_size = Size2D(3, 3);
        const auto dilation  = Size2D(0, 1);
        const bool has_bias  = false;
        const auto status    = opencl::kernels::ClIm2ColKernel::validate(&input, &output, conv_size, PadStrideInfo(), has_bias, dilation);
        ARM_COMPUTE_EXPECT(bool(status) == false, framework::LogLevel::ERRORS);
    }

    // NHWC and grouping greater than 1
    {
        const auto         input      = TensorInfo(TensorShape(10U, 12U, 1U, 2U), 1, DataType::F32, DataLayout::NHWC);
        const auto         output     = TensorInfo(TensorShape(9U, 80U, 2U), 1, DataType::F32);
        const auto         conv_size  = Size2D(3, 3);
        const auto         dilation   = Size2D(1, 1);
        const bool         has_bias   = false;
        const unsigned int num_groups = 2;
        const auto         status     = opencl::kernels::ClIm2ColKernel::validate(&input, &output, conv_size, PadStrideInfo(), has_bias, dilation, num_groups);
        ARM_COMPUTE_EXPECT(bool(status) == false, framework::LogLevel::ERRORS);
    }

    // NCWH and channels % num_groups !=0
    {
        const auto         input      = TensorInfo(TensorShape(10U, 12U, 1U, 2U), 1, DataType::F32, DataLayout::NCHW);
        const auto         output     = TensorInfo(TensorShape(9U, 80U, 2U), 1, DataType::F32);
        const auto         conv_size  = Size2D(3, 3);
        const auto         dilation   = Size2D(1, 1);
        const bool         has_bias   = false;
        const unsigned int num_groups = 2;
        const auto         status     = opencl::kernels::ClIm2ColKernel::validate(&input, &output, conv_size, PadStrideInfo(), has_bias, dilation, num_groups);
        ARM_COMPUTE_EXPECT(bool(status) == false, framework::LogLevel::ERRORS);
    }

    // Invalid output shape
    {
        const auto input     = TensorInfo(TensorShape(10U, 12U, 1U, 2U), 1, DataType::F32);
        const auto output    = TensorInfo(TensorShape(9U, 81U, 2U), 1, DataType::F32);
        const auto conv_size = Size2D(3, 3);
        const bool has_bias  = false;
        const auto status    = opencl::kernels::ClIm2ColKernel::validate(&input, &output, conv_size, PadStrideInfo(), has_bias);
        ARM_COMPUTE_EXPECT(bool(status) == false, framework::LogLevel::ERRORS);
    }

    // Kernel dimensions are too big
    {
        const auto input     = TensorInfo(TensorShape(1U, 9U, 5U, 2U), 1, DataType::F32, DataLayout::NHWC);
        const auto output    = TensorInfo(TensorShape(1U, 1U, 1U, 2U), 1, DataType::F32, DataLayout::NHWC);
        const auto conv_size = Size2D(9, 9);
        const bool has_bias  = false;
        const auto status    = opencl::kernels::ClIm2ColKernel::validate(&input, &output, conv_size, PadStrideInfo(), has_bias);
        ARM_COMPUTE_EXPECT(bool(status) == false, framework::LogLevel::ERRORS);
    }
}

template <typename T>
using ClIm2ColFixture = Im2ColOpValidationFixture<CLTensor, CLAccessor, ClIm2Col, T, true>;

TEST_SUITE(NHWC)

/** Test special kernel used for NHWC for 3x3 kernels
 *
 * @note 2 elements processed per iteration
 *
 * Three tests will be run:
 *  - Channels are multiple of elements processed
 *  - Channels larger and non multiple of elements used
 *  - Channels smaller and not multiple of elements used
 *
 *  Kernel tested im2col3x3_nhwc
 */
FIXTURE_DATA_TEST_CASE(W3x3,
                       ClIm2ColFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(
                                                                   framework::dataset::make("InputShape",
{
    TensorShape(5U, 7U, 2U, 2U), TensorShape(4U, 6U, 3U, 2U), TensorShape(5U, 3U, 1U, 2U),
}),
framework::dataset::make("DataType", DataType::F32)),
framework::dataset::make("Kernel", Size2D(3, 3))),
framework::dataset::make("PadStride", { PadStrideInfo(1, 2, 1, 2), PadStrideInfo(1, 1, 0, 0) })),
framework::dataset::make("QInfo", QuantizationInfo())),
framework::dataset::make("DataLayout", DataLayout::NHWC)),
framework::dataset::make("Groups", 1)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

/** Test special kernel used for NHWC for 9x9 kernels
 *
 * @note 2 elements processed per iteration
 *
 * Three tests will be run:
 *  - Channels are multiple of elements processed
 *  - Channels larger and non multiple of elements used
 *  - Channels smaller and not multiple of elements used
 *
 *  Kernel tested im2col9x9_nhwc
 */
FIXTURE_DATA_TEST_CASE(W9x9,
                       ClIm2ColFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(
                                                                   framework::dataset::make("InputShape",
{
    TensorShape(13U, 15U, 2U, 2U), TensorShape(15U, 12U, 3U, 2U), TensorShape(13U, 22U, 1U, 2U),
}),
framework::dataset::make("DataType", DataType::F32)),
framework::dataset::make("Kernel", Size2D(9, 9))),
framework::dataset::make("PadStride", { PadStrideInfo(2, 2, 1, 2), PadStrideInfo(1, 1, 0, 0) })),
framework::dataset::make("QInfo", QuantizationInfo())),
framework::dataset::make("DataLayout", DataLayout::NHWC)),
framework::dataset::make("Groups", 1)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

/** Test generic kernel used for NHWC
 *
 * @note 2 elements processed per iteration
 *
 * Three tests will be run:
 *  - Channels are multiple of elements processed
 *  - Channels larger and non multiple of elements used
 *  - Channels smaller and not multiple of elements used
 *
 *  Kernel tested im2col_generic_nhwc
 */
FIXTURE_DATA_TEST_CASE(Generic,
                       ClIm2ColFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(
                                                                   framework::dataset::make("InputShape",
{
    TensorShape(13U, 15U, 4U, 2U), TensorShape(15U, 12U, 7U, 1U), TensorShape(5U, 3U, 1U, 1U),
}),
framework::dataset::make("DataType", DataType::F32)),
framework::dataset::make("Kernel", Size2D(5, 3))),
framework::dataset::make("PadStride", { PadStrideInfo(2, 2, 1, 2), PadStrideInfo(1, 1, 0, 0) })),
framework::dataset::make("QInfo", QuantizationInfo())),
framework::dataset::make("DataLayout", DataLayout::NHWC)),
framework::dataset::make("Groups", 1)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // NHWC

TEST_SUITE(NCHW)

/** Test special kernel used for NCHW for 1x1 kernels with stride 1 and no padding
 *
 * @note 4 elements processed per iteration
 *
 * Three tests will be run:
 *  - Channels are multiple of elements processed
 *  - Channels larger and non multiple of elements used
 *  - Channels smaller and not multiple of elements used
 *
 *  Kernel tested im2col1x1_stridex1_nchw
 */
FIXTURE_DATA_TEST_CASE(W1x1_Stride1_NoPad,
                       ClIm2ColFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(
                                                                   framework::dataset::make("InputShape", { TensorShape(4U, 4U, 3U, 2U), TensorShape(5U, 4U, 3U, 2U), TensorShape(3U, 4U, 3U, 2U) }),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                               framework::dataset::make("Kernel", Size2D(1, 1))),
                                                       framework::dataset::make("PadStride", PadStrideInfo(1, 1, 0, 0))),
                                               framework::dataset::make("QInfo", QuantizationInfo())),
                                       framework::dataset::make("DataLayout", DataLayout::NCHW)),
                               framework::dataset::make("Groups", 1)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

/** Test special kernel used for NCHW for 3x3 kernels
 *
 * @note 1 elements processed per iteration
 *
 * Executed single test as padding is required.
 *
 *  Kernel tested im2col3x3_nchw
 */
FIXTURE_DATA_TEST_CASE(W3x3,
                       ClIm2ColFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(
                                                                   framework::dataset::make("InputShape", TensorShape(4U, 4U, 3U, 2U)),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                               framework::dataset::make("Kernel", Size2D(3, 3))),
                                                       framework::dataset::make("PadStride", PadStrideInfo(1, 2, 1, 2))),
                                               framework::dataset::make("QInfo", QuantizationInfo())),
                                       framework::dataset::make("DataLayout", DataLayout::NCHW)),
                               framework::dataset::make("Groups", { 1, 3 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

/** Test special kernel used for NCHW for 5x5 kernels
 *
 * @note 1 elements processed per iteration
 *
 * Executed single test as padding is required.
 *
 *  Kernel tested im2col5x5_nchw
 */
FIXTURE_DATA_TEST_CASE(W5x5,
                       ClIm2ColFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(
                                                                   framework::dataset::make("InputShape", TensorShape(7U, 4U, 3U, 2U)),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                               framework::dataset::make("Kernel", Size2D(5, 5))),
                                                       framework::dataset::make("PadStride", PadStrideInfo(2, 1, 2, 1))),
                                               framework::dataset::make("QInfo", QuantizationInfo())),
                                       framework::dataset::make("DataLayout", DataLayout::NCHW)),
                               framework::dataset::make("Groups", { 1, 3 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

/** Test special kernel used for NCHW for 11x11 kernels when no padding present
 *
 * @note 1 elements processed per iteration
 *
 * Two tests will be run:
 *  - Without padding requirements
 *  - With padding requirements
 *
 * Kernel tested im2col11x11_padx0_pady0_nchw
 */
FIXTURE_DATA_TEST_CASE(W11x11_NoPad,
                       ClIm2ColFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(
                                                                   framework::dataset::make("InputShape", { TensorShape(11U, 11U, 2U, 2U), TensorShape(14U, 13U, 1U, 2U) }),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                               framework::dataset::make("Kernel", Size2D(11, 11))),
                                                       framework::dataset::make("PadStride", PadStrideInfo(1, 1, 0, 0))),
                                               framework::dataset::make("QInfo", QuantizationInfo())),
                                       framework::dataset::make("DataLayout", DataLayout::NCHW)),
                               framework::dataset::make("Groups", 1)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

/** Test special kernel used for NCHW for kernels which do not fall in the categories above and have no padding present
 *
 * @note 1 elements processed per iteration
 *
 * Executed single test as padding is required.
 *
 * Kernel tested im2col_generic_padx0_pady0_nchw
 */
FIXTURE_DATA_TEST_CASE(GenericZeroPad,
                       ClIm2ColFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(
                                                                   framework::dataset::make("InputShape", TensorShape(13U, 11U, 2U, 2U)),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                               framework::dataset::make("Kernel", Size2D(3, 2))),
                                                       framework::dataset::make("PadStride", PadStrideInfo(2, 1, 0, 0))),
                                               framework::dataset::make("QInfo", QuantizationInfo())),
                                       framework::dataset::make("DataLayout", DataLayout::NCHW)),
                               framework::dataset::make("Groups", { 1, 2 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // NCHW

/** Generic NCHW/NHWC kernel
 *
 * @note 1 elements processed per iteration
 *
 * Padding is not needed thus executed sample tests with different kernels sizes
 * and stride/padding information
 *
 * Kernel tested im2col_generic_(nchw|nhwc)
 */
FIXTURE_DATA_TEST_CASE(Generic,
                       ClIm2ColFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(
                                                                   framework::dataset::make("InputShape", TensorShape(13U, 11U, 5U, 2U)),
                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                               framework::dataset::make("Kernel", { Size2D(3, 2), Size2D(3, 5) })),
                                                       framework::dataset::make("PadStride", PadStrideInfo(2, 1, 2, 1))),
                                               framework::dataset::make("QInfo", QuantizationInfo())),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               framework::dataset::make("Groups", 1)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

/** Tests to check that quantized padding value is set correctly
 *
 * Kernels tested:
 *  - im2col_generic_nhwc
 *  - im2col_generic_nchw
 *  - im2col5x5_nchw
 *  - im2col3x3_nhwc
 *  - im2col3x3_nchw
 *  - im2col9x9_nhwc
 */
FIXTURE_DATA_TEST_CASE(Quantized,
                       ClIm2ColFixture<uint8_t>,
                       framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(
                                                                   framework::dataset::make("InputShape", TensorShape(13U, 11U, 11U, 2U)),
                                                                   framework::dataset::make("DataType", DataType::QASYMM8)),
                                                               framework::dataset::make("Kernel", { Size2D(1, 1), Size2D(3, 3), Size2D(5, 5), Size2D(3, 5), Size2D(9, 9) })),
                                                       framework::dataset::make("PadStride", { PadStrideInfo(1, 2, 1, 1) })),
                                               framework::dataset::make("QInfo", QuantizationInfo(0.5f, 10))),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               framework::dataset::make("Groups", 1)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

/** Tests to check that half-precision execution
 *
 * Kernels tested:
 *  - im2col_generic_nhwc
 *  - im2col_generic_nchw
 *  - im2col5x5_nchw
 *  - im2col3x3_nhwc
 *  - im2col3x3_nchw
 *  - im2col9x9_nhwc
 */
FIXTURE_DATA_TEST_CASE(FP16,
                       ClIm2ColFixture<half>,
                       framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(
                                                                   framework::dataset::make("InputShape", TensorShape(13U, 11U, 11U, 2U)),
                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                               framework::dataset::make("Kernel", { Size2D(1, 1), Size2D(3, 3), Size2D(5, 5), Size2D(3, 5), Size2D(9, 9) })),
                                                       framework::dataset::make("PadStride", { PadStrideInfo(1, 2, 1, 1) })),
                                               framework::dataset::make("QInfo", QuantizationInfo())),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               framework::dataset::make("Groups", 1)))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

TEST_SUITE_END() // Im2Col
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
