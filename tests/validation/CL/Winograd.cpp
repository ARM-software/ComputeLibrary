/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLWinogradFilterTransformKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLWinogradInputTransform.h"
#include "tests/CL/CLAccessor.h"
#include "tests/CL/Helper.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/datasets/WinogradFilterTransformDataset.h"
#include "tests/datasets/WinogradInputTransformDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/WinogradLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr AbsoluteTolerance<float> tolerance_f32(0.0001f);
} // namespace

using namespace arm_compute::misc::shape_calculator;

TEST_SUITE(CL)
TEST_SUITE(Winograd)

TEST_SUITE(InputTransform)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
                                                framework::dataset::make("InputInfo",{
                                                                                        TensorInfo(TensorShape(53U, 21U, 5U, 3U), 1, DataType::F16),     // F16 not supported
                                                                                        TensorInfo(TensorShape(53U, 21U, 5U, 3U), 1, DataType::QASYMM8), // QASYMM8 not supported
                                                                                        TensorInfo(TensorShape(53U, 21U, 5U, 3U), 1, DataType::F32),     // Kernel size not supported
                                                                                        TensorInfo(TensorShape(53U, 21U, 5U, 3U), 1, DataType::F32),     // Strides not supported
                                                                                        TensorInfo(TensorShape(53U, 33U, 4U), 1, DataType::F32),         // valid
                                                                                        TensorInfo(TensorShape(34U, 42U, 7U, 3U), 1, DataType::F32),     // valid
                                                                                        TensorInfo(TensorShape(31U, 37U, 37U), 1, DataType::F32)         // valid
                                                                                    }),
                                                framework::dataset::make("OutputInfo", {
                                                                                        TensorInfo(TensorShape(5U, 5U, 16U, 3U), 1, DataType::F16),
                                                                                        TensorInfo(TensorShape(5U, 5U, 16U, 3U), 1, DataType::QASYMM8),
                                                                                        TensorInfo(TensorShape(5U, 5U, 16U, 3U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(5U, 1U, 16U, 3U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(4U, 442U, 16U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(7U, 320U, 16U, 3U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(37U, 304U, 16U), 1, DataType::F32)
                                                                                    })),
                                                framework::dataset::make("PadStrideInfo", {
                                                                                        PadStrideInfo(1, 1, 1, 0),
                                                                                        PadStrideInfo(1, 1, 0, 0),
                                                                                        PadStrideInfo(1, 1, 1, 1),
                                                                                        PadStrideInfo(2, 1, 1, 1),
                                                                                        PadStrideInfo(1, 1, 0, 1),
                                                                                        PadStrideInfo(1, 1, 0, 0),
                                                                                        PadStrideInfo(1, 1, 1, 1)
                                                                                    })),
                                                framework::dataset::make("KernelDims", {
                                                                                        Size2D(3U, 3U),
                                                                                        Size2D(3U, 3U),
                                                                                        Size2D(5U, 5U),
                                                                                        Size2D(3U, 3U),
                                                                                        Size2D(3U, 3U),
                                                                                        Size2D(3U, 3U),
                                                                                        Size2D(3U, 3U)
                                                                                    })),
                                                framework::dataset::make("Expected", { false, false, false, false, true, true, true })),
                                            input_info, output_info, conv_info, kernel_dims, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLWinogradInputTransform::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), conv_info, kernel_dims)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

using CLWinogradInputTransformFixture = WinogradInputTransformValidationFixture<CLTensor, CLAccessor, CLWinogradInputTransform, float>;

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(framework::dataset::concat(datasets::SmallWinogradInputTransformDataset(), datasets::LargeWinogradInputTransformDataset()),
                                                                   framework::dataset::make("DataType", { DataType::F32 })),
               shape_in, conv_info, kernel_dims, is_nchw_format, data_type)
{
    ARM_COMPUTE_UNUSED(is_nchw_format);

    TensorShape shape_out = compute_winograd_input_transform_shape(TensorInfo(shape_in, 1, data_type), conv_info, kernel_dims);

    // Create tensors
    CLTensor in  = create_tensor<CLTensor>(shape_in, data_type);
    CLTensor out = create_tensor<CLTensor>(shape_out, data_type);

    ARM_COMPUTE_EXPECT(in.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(out.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLWinogradInputTransform winograd_input_transform;

    // Configure the function
    winograd_input_transform.configure(&in, &out, conv_info, kernel_dims);
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradInputTransformFixture, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallWinogradInputTransformDataset(), framework::dataset::make("DataType", { DataType::F32 })))
{
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradInputTransformFixture, framework::DatasetMode::NIGHTLY, combine(datasets::LargeWinogradInputTransformDataset(), framework::dataset::make("DataType", { DataType::F32 })))
{
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // InputTransform

TEST_SUITE(FilterTransform)
// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(
                                                framework::dataset::make("InputInfo",{
                                                                                        TensorInfo(TensorShape(3U, 3U, 5U, 3U), 1, DataType::F16),     // F16 not supported
                                                                                        TensorInfo(TensorShape(3U, 3U, 5U, 3U), 1, DataType::QASYMM8), // QASYMM8 not supported
                                                                                        TensorInfo(TensorShape(5U, 5U, 5U, 3U), 1, DataType::F32),     // Kernel size not supported
                                                                                        TensorInfo(TensorShape(3U, 3U), 1, DataType::F32),             // valid
                                                                                        TensorInfo(TensorShape(3U, 3U, 5U, 3U), 1, DataType::F32),     // valid
                                                                                        TensorInfo(TensorShape(3U, 3U, 37U, 2U), 1, DataType::F32),    // valid
                                                                                        TensorInfo(TensorShape(3U, 3U, 37U, 22U), 1, DataType::F32)    // valid
                                                                                    }),
                                                framework::dataset::make("OutputInfo", {
                                                                                        TensorInfo(TensorShape(3U, 5U, 16U), 1, DataType::F16),
                                                                                        TensorInfo(TensorShape(3U, 5U, 16U), 1, DataType::QASYMM8),
                                                                                        TensorInfo(TensorShape(3U, 5U, 16U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(1U, 1U, 16U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(3U, 5U, 16U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(2U, 37U, 16U), 1, DataType::F32),
                                                                                        TensorInfo(TensorShape(22U, 37U, 16U), 1, DataType::F32)
                                                                                    })),
                                                framework::dataset::make("Expected", { false, false, false, true, true, true, true })),
                                            input_info, output_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLWinogradFilterTransformKernel::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false))) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

using CLWinogradFilterTransform        = CLSynthetizeFunctionWithZeroConstantBorder<CLWinogradFilterTransformKernel, 0>;
using CLWinogradFilterTransformFixture = WinogradFilterTransformValidationFixture<CLTensor, CLAccessor, CLWinogradFilterTransform, float>;

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(framework::dataset::concat(datasets::SmallWinogradFilterTransformDataset(), datasets::LargeWinogradFilterTransformDataset()),
                                                                   framework::dataset::make("DataType", { DataType::F32 })),
               shape_a, is_nchw_format, data_type)
{
    ARM_COMPUTE_UNUSED(is_nchw_format);

    TensorShape shape_b = compute_winograd_filter_transform_shape(TensorInfo(shape_a, 1, data_type));

    // Create tensors
    CLTensor a = create_tensor<CLTensor>(shape_a, data_type);
    CLTensor b = create_tensor<CLTensor>(shape_b, data_type);

    ARM_COMPUTE_EXPECT(a.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(b.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLWinogradFilterTransform winograd_filter_transform;
    winograd_filter_transform.configure(&a, &b);
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLWinogradFilterTransformFixture, framework::DatasetMode::ALL, combine(datasets::SmallWinogradFilterTransformDataset(), framework::dataset::make("DataType", { DataType::F32 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLWinogradFilterTransformFixture, framework::DatasetMode::NIGHTLY, combine(datasets::LargeWinogradFilterTransformDataset(), framework::dataset::make("DataType", { DataType::F32 })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // FilterTransform

TEST_SUITE_END() // Winograd
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
