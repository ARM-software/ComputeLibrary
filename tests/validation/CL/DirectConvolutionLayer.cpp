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
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/DirectConvolutionLayerDataset.h"
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
// COMPMID-517 Invesitgate the mismatch to see whether it is a real bug
RelativeTolerance<half>  tolerance_fp16(half(0.2)); /**< Tolerance for floating point tests */
RelativeTolerance<float> tolerance_fp32(0.02f);     /**< Tolerance for floating point tests */
constexpr float          tolerance_num = 0.07f;     /**< Tolerance number */

constexpr AbsoluteTolerance<int8_t>  tolerance_qs8(0);     /**< Tolerance for fixed point tests */
constexpr AbsoluteTolerance<int16_t> tolerance_qs16(0);    /**< Tolerance for fixed point tests */
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(1); /**< Tolerance for quantized tests */

/** Direct convolution data set. */
const auto data = combine(datasets::SmallDirectConvolutionShapes(),
                          combine(framework::dataset::make("StrideX", 1, 3),
                                  combine(framework::dataset::make("StrideY", 1, 3),
                                          combine(concat(combine(framework::dataset::make("PadX", 0),
                                                                 combine(framework::dataset::make("PadY", 0),
                                                                         framework::dataset::make("KernelSize", 1))),
                                                         combine(framework::dataset::make("PadX", 0, 2),
                                                                 combine(framework::dataset::make("PadY", 0, 2),
                                                                         framework::dataset::make("KernelSize", { 3, 5 })))),
                                                  framework::dataset::make("NumKernels", { 1, 4, 8, 16 })))));
const auto data_fixed_point = combine(datasets::SmallDirectConvolutionShapes(),
                                      combine(framework::dataset::make("StrideX", 1, 3),
                                              combine(framework::dataset::make("StrideY", 1, 3),
                                                      combine(concat(combine(framework::dataset::make("PadX", 0),
                                                                             combine(framework::dataset::make("PadY", 0),
                                                                                     framework::dataset::make("KernelSize", 1))),
                                                                     combine(framework::dataset::make("PadX", 0, 2),
                                                                             combine(framework::dataset::make("PadY", 0, 2),
                                                                                     framework::dataset::make("KernelSize", { 3 })))),
                                                              framework::dataset::make("NumKernels", { 1, 4, 8, 16 })))));
} // namespace

TEST_SUITE(CL)
TEST_SUITE(DirectConvolutionLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, 0), // Mismatching data type input/weights
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, 0), // Mismatching input feature maps
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, 0), // Unsupported kernel width
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, 0), // Non-rectangular weights dimensions
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, 0), // Invalid weights dimensions
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, 0), // Invalid stride
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, 0), // Invalid biases size
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, 0), // Invalid biases dimensions
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, 0), // Invalid output size
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, 0), // Window shrink
                                                       TensorInfo(TensorShape(32U, 16U, 2U), 1, DataType::F32, 0),
                                                     }),
               framework::dataset::make("WeightsInfo",{ TensorInfo(TensorShape(3U, 3U, 2U, 4U), 1, DataType::F16, 0),
                                                        TensorInfo(TensorShape(3U, 3U, 3U, 4U), 1, DataType::F32, 0),
                                                        TensorInfo(TensorShape(9U, 9U, 2U, 4U), 1, DataType::F32, 0),
                                                        TensorInfo(TensorShape(5U, 3U, 2U, 4U), 1, DataType::F32, 0),
                                                        TensorInfo(TensorShape(3U, 3U, 2U, 4U, 3U), 1, DataType::F32, 0),
                                                        TensorInfo(TensorShape(3U, 3U, 2U, 4U), 1, DataType::F32, 0),
                                                        TensorInfo(TensorShape(3U, 3U, 2U, 4U), 1, DataType::F32, 0),
                                                        TensorInfo(TensorShape(3U, 3U, 2U, 4U), 1, DataType::F32, 0),
                                                        TensorInfo(TensorShape(3U, 3U, 2U, 4U), 1, DataType::F32, 0),
                                                        TensorInfo(TensorShape(3U, 3U, 2U, 4U), 1, DataType::F32, 0),
                                                        TensorInfo(TensorShape(1U, 1U, 2U, 4U), 1, DataType::F32, 0),
                                                     })),
               framework::dataset::make("BiasesInfo",{ TensorInfo(TensorShape(4U), 1, DataType::F32, 0),
                                                       TensorInfo(TensorShape(4U), 1, DataType::F32, 0),
                                                       TensorInfo(TensorShape(4U), 1, DataType::F32, 0),
                                                       TensorInfo(TensorShape(4U), 1, DataType::F32, 0),
                                                       TensorInfo(TensorShape(4U), 1, DataType::F32, 0),
                                                       TensorInfo(TensorShape(4U), 1, DataType::F32, 0),
                                                       TensorInfo(TensorShape(3U), 1, DataType::F32, 0),
                                                       TensorInfo(TensorShape(4U, 2U), 1, DataType::F32, 0),
                                                       TensorInfo(TensorShape(4U), 1, DataType::F32, 0),
                                                       TensorInfo(TensorShape(4U), 1, DataType::F32, 0),
                                                       TensorInfo(TensorShape(4U), 1, DataType::F32, 0),
                                                     })),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(25U, 11U, 4U), 1, DataType::F32, 0),
                                                       TensorInfo(TensorShape(25U, 11U, 4U), 1, DataType::F32, 0),
                                                       TensorInfo(TensorShape(25U, 11U, 4U), 1, DataType::F32, 0),
                                                       TensorInfo(TensorShape(25U, 11U, 4U), 1, DataType::F32, 0),
                                                       TensorInfo(TensorShape(25U, 11U, 4U), 1, DataType::F32, 0),
                                                       TensorInfo(TensorShape(25U, 11U, 4U), 1, DataType::F32, 0),
                                                       TensorInfo(TensorShape(25U, 11U, 4U), 1, DataType::F32, 0),
                                                       TensorInfo(TensorShape(25U, 11U, 4U), 1, DataType::F32, 0),
                                                       TensorInfo(TensorShape(26U, 11U, 4U), 1, DataType::F32, 0),
                                                       TensorInfo(TensorShape(25U, 11U, 4U), 1, DataType::F32, 0),
                                                       TensorInfo(TensorShape(32U, 16U, 4U), 1, DataType::F32, 0),
                                                     })),
               framework::dataset::make("ConvInfo",  { PadStrideInfo(1, 1, 0, 0),
                                                       PadStrideInfo(1, 1, 0, 0),
                                                       PadStrideInfo(1, 1, 0, 0),
                                                       PadStrideInfo(1, 1, 0, 0),
                                                       PadStrideInfo(1, 1, 0, 0),
                                                       PadStrideInfo(3, 3, 0, 0),
                                                       PadStrideInfo(1, 1, 0, 0),
                                                       PadStrideInfo(1, 1, 0, 0),
                                                       PadStrideInfo(1, 1, 0, 0),
                                                       PadStrideInfo(1, 1, 0, 0),
                                                       PadStrideInfo(1, 1, 0, 0),
                                                      })),
               framework::dataset::make("Expected", { false, false, false, false, false, false, false, false, false, false, true })),
               input_info, weights_info, biases_info, output_info, conv_info, expected)
{
    bool is_valid = bool(CLDirectConvolutionLayer::validate(&input_info.clone()->set_is_resizable(false), &weights_info.clone()->set_is_resizable(false), &biases_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), conv_info));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLDirectConvolutionLayerFixture = DirectConvolutionValidationFixture<CLTensor, CLAccessor, CLDirectConvolutionLayer, T>;
template <typename T>
using CLDirectConvolutionValidationWithTensorShapesFixture = DirectConvolutionValidationWithTensorShapesFixture<CLTensor, CLAccessor, CLDirectConvolutionLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(Run, CLDirectConvolutionLayerFixture<half>, framework::DatasetMode::ALL, combine(data, framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp16, tolerance_num);
}
TEST_SUITE_END()

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(Run, CLDirectConvolutionLayerFixture<float>, framework::DatasetMode::ALL, combine(data, framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp32);
}
TEST_SUITE_END()

TEST_SUITE(FP32_CustomDataset)
FIXTURE_DATA_TEST_CASE(Run, CLDirectConvolutionValidationWithTensorShapesFixture<float>, framework::DatasetMode::ALL, combine(datasets::DirectConvolutionLayerDataset(),
                       framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_fp32);
}
TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using CLDirectConvolutionLayerFixedPointFixture = DirectConvolutionValidationFixedPointFixture<CLTensor, CLAccessor, CLDirectConvolutionLayer, T>;

TEST_SUITE(FixedPoint)
TEST_SUITE(QS8)
FIXTURE_DATA_TEST_CASE(Run, CLDirectConvolutionLayerFixedPointFixture<int8_t>, framework::DatasetMode::ALL, combine(combine(data_fixed_point, framework::dataset::make("DataType", DataType::QS8)),
                                                                                                                    framework::dataset::make("FractionalBits", 2, 7)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qs8);
}
TEST_SUITE_END()

TEST_SUITE(QS16)
FIXTURE_DATA_TEST_CASE(Run, CLDirectConvolutionLayerFixedPointFixture<int16_t>, framework::DatasetMode::ALL, combine(combine(data_fixed_point, framework::dataset::make("DataType", DataType::QS16)),
                                                                                                                     framework::dataset::make("FractionalBits", 2, 15)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qs16);
}
TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using CLDirectConvolutionLayerQuantizedFixture = DirectConvolutionValidationQuantizedFixture<CLTensor, CLAccessor, CLDirectConvolutionLayer, T>;
template <typename T>
using CLDirectConvolutionValidationWithTensorShapesQuantizedFixture = DirectConvolutionValidationWithTensorShapesQuantizedFixture<CLTensor, CLAccessor, CLDirectConvolutionLayer, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(Run, CLDirectConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::ALL, combine(combine(data, framework::dataset::make("DataType", DataType::QASYMM8)),
                                                                                                                    framework::dataset::make("QuantizationInfo", { QuantizationInfo(2.f / 255, 10) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END()

TEST_SUITE(QASYMM8_CustomDataset)
FIXTURE_DATA_TEST_CASE(Run, CLDirectConvolutionValidationWithTensorShapesQuantizedFixture<uint8_t>, framework::DatasetMode::ALL, combine(combine(datasets::DirectConvolutionLayerDataset(),
                       framework::dataset::make("DataType", DataType::QASYMM8)),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(2.f / 255, 127) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
