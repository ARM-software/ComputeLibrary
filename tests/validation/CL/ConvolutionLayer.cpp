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
#include "arm_compute/runtime/CL/functions/CLConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLGEMMConvolutionLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/LargeConvolutionLayerDataset.h"
#include "tests/datasets/SmallConvolutionLayerDataset.h"
#include "tests/datasets/TinyConvolutionLayerDataset.h"
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
constexpr AbsoluteTolerance<float>  absolute_tolerance_float(0.0001f);    /**< Absolute Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
RelativeTolerance<float>            tolerance_f32(0.05f);                 /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
RelativeTolerance<half_float::half> tolerance_f16(half_float::half(0.2)); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F16 */
constexpr AbsoluteTolerance<float>  tolerance_qasymm8(1);                 /**< Tolerance value for comparing reference's output against implementation's output for quantized data types */
constexpr float                     tolerance_num = 0.07f;                /**< Tolerance number */

/** CNN data types */
const auto CNNDataTypes = framework::dataset::make("DataType",
{
    DataType::F16,
    DataType::F32,
    DataType::QASYMM8,
});

/** Grouped CNN data types */
const auto GroupedCNNDataTypes = framework::dataset::make("DataType",
{
    DataType::F16,
    DataType::F32
});

const auto ActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 0.5f),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 0.5f)
});
const auto ActivationFunctionsSmallDataset = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 0.5f)
});
} // namespace

TEST_SUITE(CL)
TEST_SUITE(ConvolutionLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(ValidateConvolutionMethod, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(zip(
                                          framework::dataset::make("InputInfo", { TensorInfo(TensorShape(17U, 31U, 2U), 1, DataType::F32),      // Select GEMM
                                                                                  TensorInfo(TensorShape(17U, 31U, 2U), 1, DataType::F32),      // Select GEMM
                                                                                  TensorInfo(TensorShape(23U, 27U, 5U, 4U), 1, DataType::F32),  // Select GEMM
                                                                                  TensorInfo(TensorShape(23U, 27U, 31U, 4U), 1, DataType::F32), // Select WINOGRAD
                                                                                  TensorInfo(TensorShape(3U, 3U, 2U, 1U), 1, DataType::F32),    // Select GEMM
                                                                                  TensorInfo(TensorShape(33U, 27U, 7U, 4U), 1, DataType::F32),  // Select GEMM
                                                                                  TensorInfo(TensorShape(17U, 31U, 32U), 1, DataType::F32),     // Select WINOGRAD
                                                                                  TensorInfo(TensorShape(17U, 31U, 2U), 1, DataType::F32)       // Select GEMM
                                          }),
                                          framework::dataset::make("WeightsInfo", { TensorInfo(TensorShape(5U, 5U, 2U, 19U), 1, DataType::F32),
                                                                                    TensorInfo(TensorShape(5U, 5U, 2U, 19U), 1, DataType::F32),
                                                                                    TensorInfo(TensorShape(3U, 3U, 5U, 21U), 1, DataType::F32),
                                                                                    TensorInfo(TensorShape(3U, 3U, 31U, 21U), 1, DataType::F32),
                                                                                    TensorInfo(TensorShape(3U, 3U, 5U, 21U), 1, DataType::F32),
                                                                                    TensorInfo(TensorShape(5U, 5U, 7U, 16U), 1, DataType::F16),
                                                                                    TensorInfo(TensorShape(5U, 5U, 32U, 19U), 1, DataType::F32),
                                                                                    TensorInfo(TensorShape(5U, 5U, 2U, 19U), 1, DataType::F32)
                                          })),
                                          framework::dataset::make("OutputInfo", { TensorInfo(TensorShape(15U, 15U, 19U), 1, DataType::F32),
                                                                                   TensorInfo(TensorShape(15U, 15U, 19U), 1, DataType::F32),
                                                                                   TensorInfo(TensorShape(21U, 25U, 21U, 4U), 1, DataType::F32),
                                                                                   TensorInfo(TensorShape(21U, 25U, 21U, 4U), 1, DataType::F32),
                                                                                   TensorInfo(TensorShape(11U, 25U, 21U), 1, DataType::F32),
                                                                                   TensorInfo(TensorShape(11U, 12U, 16U, 4U), 1, DataType::F32),
                                                                                   TensorInfo(TensorShape(17U, 31U, 19U), 1, DataType::F32),
                                                                                   TensorInfo(TensorShape(17U, 31U, 19U), 1, DataType::F32)
                                          })),
                                          framework::dataset::make("ConvInfo", { PadStrideInfo(1, 2, 1, 1),
                                                                                 PadStrideInfo(1, 2, 1, 1),
                                                                                 PadStrideInfo(1, 1, 0, 0),
                                                                                 PadStrideInfo(1, 1, 0, 0),
                                                                                 PadStrideInfo(2, 1, 0, 0),
                                                                                 PadStrideInfo(3, 2, 1, 0),
                                                                                 PadStrideInfo(1, 1, 2, 2),
                                                                                 PadStrideInfo(1, 1, 2, 2)
                                          })),
                                          framework::dataset::make("GpuTarget", { GPUTarget::BIFROST,
                                                                                  GPUTarget::MIDGARD,
                                                                                  GPUTarget::G71,
                                                                                  GPUTarget::G71,
                                                                                  GPUTarget::MIDGARD,
                                                                                  GPUTarget::BIFROST,
                                                                                  GPUTarget::BIFROST,
                                                                                  GPUTarget::BIFROST
                                          })),
                                          framework::dataset::make("Dilation", { Size2D(1U, 1U),
                                                                 Size2D(1U, 1U),
                                                                 Size2D(1U, 1U),
                                                                 Size2D(1U, 1U),
                                                                 Size2D(1U, 1U),
                                                                 Size2D(1U, 1U),
                                                                 Size2D(1U, 1U),
                                                                 Size2D(2U, 1U),
                                          })),
                                         framework::dataset::make("EnableFastMath", { false, false, false, false, false, false, true, true })),
                                         framework::dataset::make("Expected",{ ConvolutionMethod::GEMM,
                                                                               ConvolutionMethod::GEMM,
                                                                               ConvolutionMethod::GEMM,
                                                                               ConvolutionMethod::WINOGRAD,
                                                                               ConvolutionMethod::GEMM,
                                                                               ConvolutionMethod::GEMM,
                                                                               ConvolutionMethod::WINOGRAD,
                                                                               ConvolutionMethod::GEMM,
                                         })),
                                         input_info, weights_info, output_info, conv_info, gpu_target, dilation, enable_fast_math, expected)
{
    ConvolutionMethod is_valid = CLConvolutionLayer::get_convolution_method(&input_info.clone()->set_is_resizable(true),
                                                                            &weights_info.clone()->set_is_resizable(true),
                                                                            &output_info.clone()->set_is_resizable(true), conv_info,
                                                                            WeightsInfo(),
                                                                            ActivationLayerInfo(),
                                                                            gpu_target,
                                                                            dilation,
                                                                            enable_fast_math);
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*
TEST_SUITE_END() // ConvolutionLayer

TEST_SUITE(GEMMConvolutionLayer)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                           CNNDataTypes),
                                                                   ActivationFunctionsDataset),
               input_shape, weights_shape, bias_shape, output_shape, info, dilation, data_type, act_info)
{
    auto bias_data_type = is_data_type_quantized_asymmetric(data_type) ? DataType::S32 : data_type;

    // Create tensors
    CLTensor src     = create_tensor<CLTensor>(input_shape, data_type, 1, QuantizationInfo(2.f / 255.f, 127));
    CLTensor weights = create_tensor<CLTensor>(weights_shape, data_type, 1, QuantizationInfo(2.f / 255.f, 127));
    CLTensor bias    = create_tensor<CLTensor>(bias_shape, bias_data_type, 1, QuantizationInfo(2.f / 255.f, 127));
    CLTensor dst     = create_tensor<CLTensor>(output_shape, data_type, 1, QuantizationInfo(2.f / 255.f, 127));

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(weights.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    const QuantizationInfo src_quantization_info     = src.info()->quantization_info();
    const QuantizationInfo weights_quantization_info = weights.info()->quantization_info();

    // Create and configure function
    CLGEMMConvolutionLayer conv;
    conv.configure(&src, &weights, &bias, &dst, info, WeightsInfo(), dilation, act_info);

    // Validate valid region
    const ValidRegion src_valid_region     = shape_to_valid_region(input_shape);
    const ValidRegion weights_valid_region = shape_to_valid_region(weights_shape);
    const ValidRegion bias_valid_region    = shape_to_valid_region(bias_shape);
    const ValidRegion dst_valid_region     = shape_to_valid_region(output_shape);

    validate(src.info()->valid_region(), src_valid_region);
    validate(weights.info()->valid_region(), weights_valid_region);
    validate(bias.info()->valid_region(), bias_valid_region);
    validate(dst.info()->valid_region(), dst_valid_region);

    // Validate QuantizationInfo
    ARM_COMPUTE_EXPECT(src.info()->quantization_info() == src_quantization_info, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(weights.info()->quantization_info() == weights_quantization_info, framework::LogLevel::ERRORS);

    // Validate padding
    //TODO(COMPMID-415) Need to validate padding?
}

template <typename T>
using CLGEMMConvolutionLayerFixture = ConvolutionValidationFixture<CLTensor, CLAccessor, CLGEMMConvolutionLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMConvolutionLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallConvolutionLayerReducedDataset(),
                                                                                                                 framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                 framework::dataset::make("DataType",
                                                                                                                         DataType::F16)),
                                                                                                                 framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                 ActivationFunctionsSmallDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMConvolutionLayerFixture<half>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(framework::dataset::concat(datasets::SmallConvolutionLayerDataset(), datasets::LargeConvolutionLayerDataset()),
                                                       framework::dataset::make("ReshapeWeights", { true })),
                                               framework::dataset::make("DataType",
                                                                        DataType::F16)),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
TEST_SUITE_END() // FP16

TEST_SUITE(FP32)

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallConvolutionLayerReducedDataset(),
                                                                                                                  framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                  framework::dataset::make("DataType",
                                                                                                                          DataType::F32)),
                                                                                                                  framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                                  ActivationFunctionsSmallDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(framework::dataset::concat(datasets::SmallConvolutionLayerDataset(), datasets::LargeConvolutionLayerDataset()),
                                                       framework::dataset::make("ReshapeWeights", { true })),
                                               framework::dataset::make("DataType",
                                                                        DataType::F32)),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32, 0.f, absolute_tolerance_float);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

template <typename T>
using CLGEMMConvolutionLayerQuantizedFixture = ConvolutionValidationQuantizedFixture<CLTensor, CLAccessor, CLGEMMConvolutionLayer, T>;

const auto QuantizedActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f)
});
const auto QuantizedActivationFunctionsSmallDataset = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f)
});

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)

const auto QuantizationData = framework::dataset::make("QuantizationInfo",
{
    QuantizationInfo(0.5f, 10),
    QuantizationInfo(0.3f, 3),
    QuantizationInfo(1.f, 10),
});

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT,
                       combine(combine(combine(combine(combine(datasets::SmallConvolutionLayerReducedDataset(),
                                                               framework::dataset::make("ReshapeWeights", { true })),
                                                       framework::dataset::make("DataType", DataType::QASYMM8)),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                       QuantizationData),
                               QuantizedActivationFunctionsSmallDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(combine(framework::dataset::concat(datasets::SmallConvolutionLayerDataset(), datasets::LargeConvolutionLayerDataset()),
                                                               framework::dataset::make("ReshapeWeights", { true })),
                                                       framework::dataset::make("DataType", DataType::QASYMM8)),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                       QuantizationData),
                               QuantizedActivationFunctionsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // GEMMConvolutionLayer

template <typename T>
using CLGEMMGroupedConvolutionLayerFixture = ConvolutionValidationFixture<CLTensor, CLAccessor, CLGEMMConvolutionLayer, T>;

TEST_SUITE(GroupedGEMMConvolutionLayer)

DATA_TEST_CASE(Configuration, framework::DatasetMode::ALL, combine(combine(datasets::SmallGroupedConvolutionLayerDataset(),
                                                                           GroupedCNNDataTypes),
                                                                   ActivationFunctionsDataset),
               input_shape, weights_shape, bias_shape, output_shape, info, dilation, data_type, act_info)
{
    ARM_COMPUTE_ERROR_ON((input_shape[2] % weights_shape[2]) != 0);

    // The number of groups is calculated dividing the number of input channels of the input tensor by the number of input channels of the weights shape
    const int num_groups = input_shape[2] / weights_shape[2];

    // Create tensors
    CLTensor src     = create_tensor<CLTensor>(input_shape, data_type);
    CLTensor weights = create_tensor<CLTensor>(weights_shape, data_type, 1);
    CLTensor bias    = create_tensor<CLTensor>(bias_shape, data_type, 1);
    CLTensor dst     = create_tensor<CLTensor>(output_shape, data_type, 1);

    ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(weights.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

    // Create and configure function
    CLGEMMConvolutionLayer conv;
    conv.configure(&src, &weights, &bias, &dst, info, WeightsInfo(), dilation, act_info, num_groups);

    // Validate valid region
    const ValidRegion src_valid_region     = shape_to_valid_region(input_shape);
    const ValidRegion weights_valid_region = shape_to_valid_region(weights_shape);
    const ValidRegion bias_valid_region    = shape_to_valid_region(bias_shape);
    const ValidRegion dst_valid_region     = shape_to_valid_region(output_shape);

    validate(src.info()->valid_region(), src_valid_region);
    validate(weights.info()->valid_region(), weights_valid_region);
    validate(bias.info()->valid_region(), bias_valid_region);
    validate(dst.info()->valid_region(), dst_valid_region);

    // Validate padding
    //TODO(COMPMID-415) Need to validate padding?
}

TEST_SUITE(Float)
TEST_SUITE(FP32)

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMGroupedConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallGroupedConvolutionLayerDataset(),
                       framework::dataset::make("ReshapeWeights", { true })),
                       framework::dataset::make("DataType", DataType::F32)),
                       framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                       ActivationFunctionsSmallDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32, tolerance_num);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMGroupedConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(framework::dataset::concat(datasets::SmallGroupedConvolutionLayerDataset(), datasets::LargeGroupedConvolutionLayerDataset()),
                                                       framework::dataset::make("ReshapeWeights", { true })),
                                               framework::dataset::make("DataType", DataType::F32)),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                               ActivationFunctionsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32, tolerance_num);
}
TEST_SUITE_END() // FP32

TEST_SUITE(FP16)

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMGroupedConvolutionLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallGroupedConvolutionLayerDataset(),
                                                                                                                        framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                        framework::dataset::make("DataType", DataType::F16)),
                                                                                                                        framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                                                                                                        ActivationFunctionsSmallDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32, tolerance_num);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMGroupedConvolutionLayerFixture<half>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(framework::dataset::concat(datasets::SmallGroupedConvolutionLayerDataset(), datasets::LargeGroupedConvolutionLayerDataset()),
                                                       framework::dataset::make("ReshapeWeights", { true })),
                                               framework::dataset::make("DataType", DataType::F16)),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                               ActivationFunctionsDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32, tolerance_num);
}
TEST_SUITE_END() // FP16
TEST_SUITE_END() // Float

TEST_SUITE_END() // GroupedGEMMConvolutionLayer
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
