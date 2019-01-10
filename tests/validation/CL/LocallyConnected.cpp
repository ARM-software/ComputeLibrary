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
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLLocallyConnectedLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/LocallyConnectedDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/LocallyConnectedFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr AbsoluteTolerance<float> atolerance_f32(0.00001f); /**< Absolute Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
RelativeTolerance<float>           rtolerance_f32(0.05f);    /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
} // namespace

TEST_SUITE(CL)
TEST_SUITE(LocallyConnected)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(
    framework::dataset::make("InputInfo",  { TensorInfo(TensorShape(23U, 27U, 5U), 1, DataType::F32), // Mismatching data type input/weights
                                             TensorInfo(TensorShape(23U, 27U, 5U), 1, DataType::F32), // Mismatching data type input/bias
                                             TensorInfo(TensorShape(23U, 27U, 5U), 1, DataType::F32), // Mismatching data type input/output
                                             TensorInfo(TensorShape(23U, 27U, 5U), 1, DataType::F32), // Mismatching shape input/weights
                                             TensorInfo(TensorShape(23U, 27U, 5U), 1, DataType::F32), // Mismatching shape input/bias
                                             TensorInfo(TensorShape(23U, 27U, 5U), 1, DataType::F32), // Mismatching shape input/output
                                             TensorInfo(TensorShape(23U, 27U, 5U), 1, DataType::F32), // Asymmetric padding
                                             TensorInfo(TensorShape(23U, 27U, 5U), 1, DataType::F32), // Padding required
                                             TensorInfo(TensorShape(23U, 27U, 5U), 1, DataType::F32)
                                           }),
    framework::dataset::make("WeightsInfo",{ TensorInfo(TensorShape(3U, 3U, 5U, 21U, 275U), 1, DataType::F16),
                                             TensorInfo(TensorShape(3U, 3U, 5U, 21U, 275U), 1, DataType::F32),
                                             TensorInfo(TensorShape(3U, 3U, 5U, 21U, 275U), 1, DataType::F32),
                                             TensorInfo(TensorShape(3U, 3U, 5U, 21U, 274U), 1, DataType::F32),
                                             TensorInfo(TensorShape(3U, 3U, 5U, 21U, 275U), 1, DataType::F32),
                                             TensorInfo(TensorShape(3U, 3U, 5U, 21U, 275U), 1, DataType::F32),
                                             TensorInfo(TensorShape(3U, 3U, 5U, 21U, 275U), 1, DataType::F32),
                                             TensorInfo(TensorShape(3U, 3U, 5U, 21U, 275U), 1, DataType::F32),
                                             TensorInfo(TensorShape(1U, 3U, 5U, 21U, 575U), 1, DataType::F32)
                                           })),
    framework::dataset::make("BiasInfo",   { TensorInfo(TensorShape(21U, 275U), 1, DataType::F32),
                                             TensorInfo(TensorShape(21U, 275U), 1, DataType::F16),
                                             TensorInfo(TensorShape(21U, 275U), 1, DataType::F32),
                                             TensorInfo(TensorShape(21U, 275U), 1, DataType::F32),
                                             TensorInfo(TensorShape(21U, 274U), 1, DataType::F32),
                                             TensorInfo(TensorShape(21U, 275U), 1, DataType::F32),
                                             TensorInfo(TensorShape(21U, 275U), 1, DataType::F32),
                                             TensorInfo(TensorShape(21U, 275U), 1, DataType::F32),
                                             TensorInfo(TensorShape(21U, 575U), 1, DataType::F32)
                                           })),
    framework::dataset::make("OutputInfo", { TensorInfo(TensorShape(11U, 25U, 21U), 1, DataType::F32),
                                             TensorInfo(TensorShape(11U, 25U, 21U), 1, DataType::F32),
                                             TensorInfo(TensorShape(11U, 25U, 21U), 1, DataType::F16),
                                             TensorInfo(TensorShape(11U, 25U, 21U), 1, DataType::F32),
                                             TensorInfo(TensorShape(11U, 25U, 21U), 1, DataType::F32),
                                             TensorInfo(TensorShape(11U, 25U, 22U), 1, DataType::F32),
                                             TensorInfo(TensorShape(11U, 25U, 21U), 1, DataType::F32),
                                             TensorInfo(TensorShape(11U, 25U, 21U), 1, DataType::F32),
                                             TensorInfo(TensorShape(23U, 25U, 21U), 1, DataType::F32)
                                           })),
    framework::dataset::make("PadStride",  { PadStrideInfo(2, 1, 0, 0),
                                             PadStrideInfo(2, 1, 0, 0),
                                             PadStrideInfo(2, 1, 0, 0),
                                             PadStrideInfo(2, 1, 0, 0),
                                             PadStrideInfo(2, 1, 0, 0),
                                             PadStrideInfo(2, 1, 0, 0),
                                             PadStrideInfo(2, 1, 1, 0),
                                             PadStrideInfo(2, 1, 0, 0),
                                             PadStrideInfo(1, 1, 0, 0)
                                           })),
    framework::dataset::make("Expected", { false, false, false, false, false, false, false, false, true })),
    input_info, weights_info, bias_info, output_info, conv_info, expected)
{
    bool is_valid = bool(CLLocallyConnectedLayer::validate(&input_info.clone()->set_is_resizable(false),
                                                           &weights_info.clone()->set_is_resizable(false),
                                                           &bias_info.clone()->set_is_resizable(false),
                                                           &output_info.clone()->set_is_resizable(false),
                                                           conv_info));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(datasets::SmallLocallyConnectedDataset(),
                                                                   framework::dataset::make("DataType", DataType::F32)),
               src_shape, weights_shape, bias_shape, dst_shape, info, dilation, data_type)
{
    ARM_COMPUTE_UNUSED(dilation);

    // Create tensors
    CLTensor src     = create_tensor<CLTensor>(src_shape, data_type);
    CLTensor weights = create_tensor<CLTensor>(weights_shape, data_type);
    CLTensor bias    = create_tensor<CLTensor>(bias_shape, data_type);
    CLTensor dst     = create_tensor<CLTensor>(dst_shape, data_type);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(weights.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function.
    CLLocallyConnectedLayer lc;
    lc.configure(&src, &weights, &bias, &dst, info);

    // Validate valid region
    const ValidRegion dst_valid_region = shape_to_valid_region(dst_shape);
    validate(dst.info()->valid_region(), dst_valid_region);
}

template <typename T>
using CLLocallyConnectedFixture = LocallyConnectedValidationFixture<CLTensor, CLAccessor, CLLocallyConnectedLayer, T>;
FIXTURE_DATA_TEST_CASE(RunSmall, CLLocallyConnectedFixture<float>, framework::DatasetMode::PRECOMMIT, combine(datasets::SmallLocallyConnectedDataset(),
                                                                                                              framework::dataset::make("DataType",
                                                                                                                      DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rtolerance_f32, 0.f, atolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLLocallyConnectedFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeLocallyConnectedDataset(),
                                                                                                            framework::dataset::make("DataType",
                                                                                                                    DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rtolerance_f32, 0.f, atolerance_f32);
}
TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
