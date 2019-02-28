/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLIm2ColKernel.h"
#include "arm_compute/core/Types.h"
#include "tests/CL/Helper.h"

#include "tests/CL/CLAccessor.h"
#include "tests/datasets/ShapeDatasets.h"
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
namespace
{
// *INDENT-OFF*
// clang-format off
const auto conv_filter_sizes = framework::dataset::make("KernelDims", { Size2D(3U, 3U),
                                                                        Size2D(5U, 5U),
                                                                        Size2D(3U, 1U),
                                                                        Size2D(1U, 3U),
                                                                        Size2D(5U, 3U),
                                                                        Size2D(1U, 1U),
                                                                        Size2D(9U, 9U),
                                                                        Size2D(11U, 11U)} );
const auto padstrides        = framework::dataset::make("PadStride", { PadStrideInfo(1U, 1U, 0U, 0U),
                                                                       PadStrideInfo(1U, 1U, 1U, 1U),
                                                                       PadStrideInfo(2U, 2U, 0U, 2U) });
const auto conv_args         = combine(combine(combine(combine(conv_filter_sizes, padstrides),
                                                       framework::dataset::make("QuantizationInfo", QuantizationInfo(0.5f, 10))),
                                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                       framework::dataset::make("NumGroups", { 1 }));
const auto grouped_args      = combine(combine(combine(combine(conv_filter_sizes, padstrides),
                                                       framework::dataset::make("QuantizationInfo", QuantizationInfo(0.5f, 10))),
                                                       framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                                       framework::dataset::make("NumGroups", { 2, 3, 4 }));

const auto conv_filter_sizes_small = framework::dataset::make("KernelDims", { Size2D(3U, 3U),
                                                                              Size2D(3U, 1U)});
const auto padstrides_small        = framework::dataset::make("PadStride", {PadStrideInfo(2U, 2U, 0U, 2U)});
const auto conv_args_small         = combine(combine(combine(combine(conv_filter_sizes_small, padstrides_small),
                                                             framework::dataset::make("QuantizationInfo", QuantizationInfo(0.5f, 10))),
                                                             framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                             framework::dataset::make("NumGroups", { 1 }));
const auto grouped_args_small      = combine(combine(combine(combine(conv_filter_sizes_small, padstrides_small),
                                                            framework::dataset::make("QuantizationInfo", QuantizationInfo(0.5f, 10))),
                                                            framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                                            framework::dataset::make("NumGroups", { 2 }));
} // namespace
TEST_SUITE(CL)
TEST_SUITE(Im2Col)

using CLIm2Col = CLSynthetizeFunction<CLIm2ColKernel>;

DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(10U, 12U, 2U), 1, DataType::U8),      // Unsupported data type
                                                       TensorInfo(TensorShape(10U, 12U, 2U), 1, DataType::F32),     // Mismatching data type
                                                       TensorInfo(TensorShape(10U, 12U, 2U), 1, DataType::QASYMM8), // Bias not supported with QASYMM8
                                                       TensorInfo(TensorShape(10U, 12U, 2U, 2U), 1, DataType::QASYMM8),
                                                     }),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(3U, 4U, 10U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(3U, 4U, 10U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(3U, 3U, 10U, 2U), 1, DataType::QASYMM8),
                                                       TensorInfo(TensorShape(18U, 80U, 2U, 1U), 1, DataType::QASYMM8),
                                                     })),
               framework::dataset::make("HasBias", { true, true, true, false })),
               framework::dataset::make("Expected", { false, false, false, true })),
               input_info, output_info, has_bias, expected)
{

    bool status = bool(CLIm2Col::validate(&input_info, &output_info, Size2D(3U, 3U), PadStrideInfo(), has_bias));
    ARM_COMPUTE_EXPECT(status == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLIm2ColFixture = Im2ColValidationFixture<CLTensor, CLAccessor, CLIm2Col, T, true>;
TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLIm2ColFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType", DataType::F32)),
                                                                                                    conv_args_small))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLIm2ColFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(concat(datasets::SmallShapes(), datasets::MediumShapes()), framework::dataset::make("DataType",
                                                                                                          DataType::F32)),
                                                                                                  conv_args))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLIm2ColFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType", DataType::F16)),
                                                                                                   conv_args_small))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLIm2ColFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(concat(datasets::SmallShapes(), datasets::MediumShapes()), framework::dataset::make("DataType",
                                                                                                         DataType::F16)),
                                                                                                 conv_args))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLIm2ColFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                      conv_args_small))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLIm2ColFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(concat(datasets::SmallShapes(), datasets::MediumShapes()),
                                                                                                            framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                    conv_args))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(Grouped)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLIm2ColFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::GroupedIm2ColSmallShapes(), framework::dataset::make("DataType",
                                                                                                            DataType::F32)),
                                                                                                    grouped_args_small))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLIm2ColFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(concat(datasets::GroupedIm2ColSmallShapes(), datasets::GroupedIm2ColLargeShapes()),
                                                                                                          framework::dataset::make("DataType",
                                                                                                                  DataType::F32)),
                                                                                                  grouped_args))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLIm2ColFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::GroupedIm2ColSmallShapes(), framework::dataset::make("DataType",
                                                                                                           DataType::F16)),
                                                                                                   grouped_args_small))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLIm2ColFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(concat(datasets::GroupedIm2ColSmallShapes(), datasets::GroupedIm2ColLargeShapes()),
                                                                                                         framework::dataset::make("DataType",
                                                                                                                 DataType::F16)),
                                                                                                 grouped_args))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLIm2ColFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::GroupedIm2ColSmallShapes(), framework::dataset::make("DataType",
                                                                                                              DataType::QASYMM8)),
                                                                                                      grouped_args_small))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLIm2ColFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(concat(datasets::GroupedIm2ColSmallShapes(), datasets::GroupedIm2ColLargeShapes()),
                                                                                                            framework::dataset::make("DataType",
                                                                                                                    DataType::QASYMM8)),
                                                                                                    grouped_args))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
