/*
 * Copyright (c) 2018-2020, 2023 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLGEMMConvolutionLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/DilatedConvolutionLayerDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ConvolutionLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
RelativeTolerance<float>            rel_tolerance_f32(0.05f);                 /**< Relative tolerance value for comparing reference's output against implementation's output for DataType::F32 */
RelativeTolerance<half_float::half> rel_tolerance_f16(half_float::half(0.2)); /**< Relative tolerance value for comparing reference's output against implementation's output for DataType::F16 */
constexpr AbsoluteTolerance<float>  abs_tolerance_qasymm8(1);                 /**< Relative tolerance value for comparing reference's output against implementation's output for quantized data types */
constexpr float                     abs_tolerance_f32 = 0.001f;               /**< Absolute tolerance value for comparing reference's output against implementation's output for DataType::F32 */
constexpr float                     abs_tolerance_f16 = 0.3f;                 /**< Absolute tolerance value for comparing reference's output against implementation's output for DataType::F16 */
constexpr float                     tolerance_num_f16 = 0.07f;                /**< Tolerance number for FP16 */

/** CNN data types */
const auto CNNDataTypes = framework::dataset::make("DataType",
{
    DataType::F16,
    DataType::F32,
    DataType::QASYMM8,
});
} // namespace

TEST_SUITE(CL)
TEST_SUITE(DilatedConvolutionLayer)

DATA_TEST_CASE(ValidateConvolutionMethod, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(
                                                                                               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(17U, 31U, 2U), 1, DataType::F32),
                                                                                                                        TensorInfo(TensorShape(17U, 31U, 2U), 1, DataType::F32),
                                                                                                                        TensorInfo(TensorShape(23U, 27U, 23U, 4U), 1, DataType::F32),
                                                                                                                        TensorInfo(TensorShape(3U, 3U, 2U, 1U), 1, DataType::F32),
                                                                                                                        TensorInfo(TensorShape(33U, 27U, 7U, 4U), 1, DataType::F32)
                                                                                                                                     }),
                                                                                               framework::dataset::make("WeightsInfo", { TensorInfo(TensorShape(5U, 5U, 2U, 19U), 1, DataType::F32),
                                                                                                                        TensorInfo(TensorShape(5U, 5U, 2U, 19U), 1, DataType::F32),
                                                                                                                        TensorInfo(TensorShape(3U, 3U, 23U, 21U), 1, DataType::F32),
                                                                                                                        TensorInfo(TensorShape(3U, 3U, 5U, 21U), 1, DataType::F32),
                                                                                                                        TensorInfo(TensorShape(5U, 5U, 7U, 16U), 1, DataType::F16)
                                                                                                                                       })),
                                                                                           framework::dataset::make("OutputInfo", { TensorInfo(TensorShape(15U, 15U, 19U), 1, DataType::F32),
                                                                                                                    TensorInfo(TensorShape(15U, 15U, 19U), 1, DataType::F32),
                                                                                                                    TensorInfo(TensorShape(21U, 25U, 21U, 4U), 1, DataType::F32),
                                                                                                                    TensorInfo(TensorShape(11U, 25U, 21U), 1, DataType::F32),
                                                                                                                    TensorInfo(TensorShape(11U, 12U, 16U, 4U), 1, DataType::F32)
                                                                                                                                  })),
                                                                                       framework::dataset::make("ConvInfo", { PadStrideInfo(1, 2, 1, 1),
                                                                                                                PadStrideInfo(1, 2, 1, 1),
                                                                                                                PadStrideInfo(1, 1, 0, 0),
                                                                                                                PadStrideInfo(2, 1, 0, 0),
                                                                                                                PadStrideInfo(3, 2, 1, 0)
                                                                                                                            })),
                                                                                   framework::dataset::make("GpuTarget", { GPUTarget::BIFROST,
                                                                                                            GPUTarget::MIDGARD,
                                                                                                            GPUTarget::G71,
                                                                                                            GPUTarget::MIDGARD,
                                                                                                            GPUTarget::BIFROST
                                                                                                                         })),
                                                                               framework::dataset::make("Dilation", { Size2D(1U, 1U),
                                                                                                                      Size2D(1U, 1U),
                                                                                                                      Size2D(1U, 1U),
                                                                                                                      Size2D(2U, 2U),
                                                                                                                      Size2D(3U, 3U)
                                                                                                                    })),

                                                                           framework::dataset::make("Expected", { ConvolutionMethod::GEMM, ConvolutionMethod::GEMM, ConvolutionMethod::WINOGRAD, ConvolutionMethod::GEMM, ConvolutionMethod::GEMM })),
               input_info, weights_info, output_info, conv_info, gpu_target, dilation, expected)
{
    ConvolutionMethod is_valid = CLConvolutionLayer::get_convolution_method(&input_info.clone()->set_is_resizable(true),
                                                                            &weights_info.clone()->set_is_resizable(true),
                                                                            &output_info.clone()->set_is_resizable(true), conv_info, WeightsInfo(), ActivationLayerInfo(), gpu_target, dilation);
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}

TEST_SUITE_END()

TEST_SUITE(GEMMDilatedConvolutionLayer)

template <typename T>
using CLGEMMDilatedConvolutionLayerFixture = ConvolutionValidationFixture<CLTensor, CLAccessor, CLGEMMConvolutionLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMDilatedConvolutionLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallDilatedConvolutionLayerDataset(),
                                                                                                                        framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                        framework::dataset::make("DataType", DataType::F16)),
                                                                                                                        framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                        framework::dataset::make("ActivationLayerInfo", ActivationLayerInfo())))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, 0.0f, abs_tolerance_f16);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMDilatedConvolutionLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeDilatedConvolutionLayerDataset(),
                                                                                                                      framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                      framework::dataset::make("DataType", DataType::F16)),
                                                                                                                      framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                      framework::dataset::make("ActivationLayerInfo", ActivationLayerInfo())))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f16, tolerance_num_f16);
}

TEST_SUITE_END()

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMDilatedConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallDilatedConvolutionLayerDataset(),
                       framework::dataset::make("ReshapeWeights", { true })),
                       framework::dataset::make("DataType", DataType::F32)),
                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                       framework::dataset::make("ActivationLayerInfo", ActivationLayerInfo())))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMDilatedConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(datasets::LargeDilatedConvolutionLayerDataset(),
                                                                                                                       framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                       framework::dataset::make("DataType", DataType::F32)),
                                                                                                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                       framework::dataset::make("ActivationLayerInfo", ActivationLayerInfo())))
{
    // Validate output
    validate(CLAccessor(_target), _reference, rel_tolerance_f32, 0.f, abs_tolerance_f32);
}
TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using CLGEMMDilatedConvolutionLayerQuantizedFixture = ConvolutionValidationQuantizedFixture<CLTensor, CLAccessor, CLGEMMConvolutionLayer, T>;

TEST_SUITE(Quantized)
/// @note: Every asymmetric quantized test where there's no fused activation will have its quantization info ignored
/// This is because instead of using the same quantization information for all the tensors, the fixture generates
/// separate quantization info for each input and the output tensor.
/// When we can also support dynamic quantization with the presence of activation, we can remove the explicit
/// quantization info.
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMDilatedConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(datasets::SmallDilatedConvolutionLayerDataset(),
                                                               framework::dataset::make("ReshapeWeights", { true })),
                                                       framework::dataset::make("DataType", DataType::QASYMM8)),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                       framework::dataset::make("IgnoredQuantizationInfo", { QuantizationInfo() })),
                               framework::dataset::make("ActivationLayerInfo", { ActivationLayerInfo() })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, abs_tolerance_qasymm8);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMDilatedConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(combine(datasets::LargeDilatedConvolutionLayerDataset(),
                                                               framework::dataset::make("ReshapeWeights", { true })),
                                                       framework::dataset::make("DataType", DataType::QASYMM8)),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                       framework::dataset::make("IgnoredQuantizationInfo", { QuantizationInfo() })),
                               framework::dataset::make("ActivationLayerInfo", { ActivationLayerInfo() })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, abs_tolerance_qasymm8);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
