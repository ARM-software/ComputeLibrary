/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLGEMMConvolutionLayer.h"
#include "src/core/experimental/PostOp.h"
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
class SmallConvolutionLayerDatasetCases final : public datasets::ConvolutionLayerDataset
{
public:
    SmallConvolutionLayerDatasetCases()
    {
        // 1D Kernel
        add_config(TensorShape(1U, 130U, 2000U), TensorShape(1U, 1U, 2000U, 2000U), TensorShape(2000U), TensorShape(1U, 130U, 2000U), PadStrideInfo(1, 1, 0, 0));
    }
};

RelativeTolerance<float>            tolerance_f32(0.1f);                  /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
RelativeTolerance<half_float::half> tolerance_f16(half_float::half(0.2)); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F16 */
constexpr AbsoluteTolerance<float>  tolerance_qasymm8(1);                 /**< Tolerance value for comparing reference's output against implementation's output for quantized data types */
constexpr float                     tolerance_num = 0.07f;                /**< Tolerance number */

/** CNN data types */
const auto CNNDataTypes = framework::dataset::make("DataType",
{
    DataType::F16,
    DataType::F32,
    DataType::QASYMM8,
    DataType::QASYMM8_SIGNED,
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

bool is_post_op_list_valid_in_gemmconv(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &output_shape, DataType data_type, DataLayout data_layout,
                                       const PadStrideInfo &conv_info, const experimental::PostOpList<ITensorInfo *> &post_ops)
{
    const int idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int idx_kernels = get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES);

    const auto         dilation   = Size2D(1U, 1U);
    const unsigned int num_groups = 1U;

    TensorInfo input_info(input_shape, 1, data_type, data_layout);
    TensorInfo weights_info(weights_shape, 1, data_type, data_layout);

    TensorInfo output_info(output_shape, 1, data_type, data_layout);

    WeightsInfo w_info(false, weights_info.dimension(idx_width), weights_info.dimension(idx_height), weights_info.dimension(idx_kernels));

    const auto status = CLGEMMConvolutionLayer::validate(&input_info.clone()->set_is_resizable(true),
                                                         &weights_info.clone()->set_is_resizable(true), nullptr, &output_info.clone()->set_is_resizable(true),
                                                         conv_info, w_info, dilation, ActivationLayerInfo(), num_groups, post_ops);
    return bool(status);
}
} // namespace

TEST_SUITE(CL)
TEST_SUITE(ConvolutionLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(ValidateConvolutionMethod, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(zip(
                                          framework::dataset::make("InputInfo", { TensorInfo(TensorShape(17U, 31U, 2U), 1, DataType::F32),            // Select GEMM
                                                                                  TensorInfo(TensorShape(17U, 31U, 2U), 1, DataType::F32),            // Select GEMM
                                                                                  TensorInfo(TensorShape(23U, 27U, 5U, 4U), 1, DataType::F32),        // Select GEMM
                                                                                  TensorInfo(TensorShape(23U, 27U, 31U, 4U), 1, DataType::F32),       // Select WINOGRAD
                                                                                  TensorInfo(TensorShape(3U, 3U, 2U, 1U), 1, DataType::F32),          // Select GEMM
                                                                                  TensorInfo(TensorShape(33U, 27U, 7U, 4U), 1, DataType::F32),        // Select GEMM
                                                                                  TensorInfo(TensorShape(17U, 31U, 32U), 1, DataType::F32),           // Select WINOGRAD
                                                                                  TensorInfo(TensorShape(17U, 31U, 2U), 1, DataType::F32),            // Select GEMM
                                                                                  TensorInfo(TensorShape(17U, 31U, 2U), 1, DataType::QASYMM8_SIGNED), // Select GEMM
                                          }),
                                          framework::dataset::make("WeightsInfo", { TensorInfo(TensorShape(5U, 5U, 2U, 19U), 1, DataType::F32),
                                                                                    TensorInfo(TensorShape(5U, 5U, 2U, 19U), 1, DataType::F32),
                                                                                    TensorInfo(TensorShape(3U, 3U, 5U, 21U), 1, DataType::F32),
                                                                                    TensorInfo(TensorShape(3U, 3U, 31U, 21U), 1, DataType::F32),
                                                                                    TensorInfo(TensorShape(3U, 3U, 5U, 21U), 1, DataType::F32),
                                                                                    TensorInfo(TensorShape(5U, 5U, 7U, 16U), 1, DataType::F16),
                                                                                    TensorInfo(TensorShape(5U, 5U, 32U, 19U), 1, DataType::F32),
                                                                                    TensorInfo(TensorShape(5U, 5U, 2U, 19U), 1, DataType::F32),
                                                                                    TensorInfo(TensorShape(5U, 5U, 2U, 19U), 1, DataType::QASYMM8_SIGNED),
                                          })),
                                          framework::dataset::make("OutputInfo", { TensorInfo(TensorShape(15U, 15U, 19U), 1, DataType::F32),
                                                                                   TensorInfo(TensorShape(15U, 15U, 19U), 1, DataType::F32),
                                                                                   TensorInfo(TensorShape(21U, 25U, 21U, 4U), 1, DataType::F32),
                                                                                   TensorInfo(TensorShape(21U, 25U, 21U, 4U), 1, DataType::F32),
                                                                                   TensorInfo(TensorShape(11U, 25U, 21U), 1, DataType::F32),
                                                                                   TensorInfo(TensorShape(11U, 12U, 16U, 4U), 1, DataType::F32),
                                                                                   TensorInfo(TensorShape(17U, 31U, 19U), 1, DataType::F32),
                                                                                   TensorInfo(TensorShape(17U, 31U, 19U), 1, DataType::F32),
                                                                                   TensorInfo(TensorShape(17U, 31U, 19U), 1, DataType::QASYMM8_SIGNED),
                                          })),
                                          framework::dataset::make("ConvInfo", { PadStrideInfo(1, 2, 1, 1),
                                                                                 PadStrideInfo(1, 2, 1, 1),
                                                                                 PadStrideInfo(1, 1, 0, 0),
                                                                                 PadStrideInfo(1, 1, 0, 0),
                                                                                 PadStrideInfo(2, 1, 0, 0),
                                                                                 PadStrideInfo(3, 2, 1, 0),
                                                                                 PadStrideInfo(1, 1, 2, 2),
                                                                                 PadStrideInfo(1, 1, 2, 2),
                                                                                 PadStrideInfo(1, 1, 2, 2),
                                          })),
                                          framework::dataset::make("GpuTarget", { GPUTarget::BIFROST,
                                                                                  GPUTarget::MIDGARD,
                                                                                  GPUTarget::G71,
                                                                                  GPUTarget::G71,
                                                                                  GPUTarget::MIDGARD,
                                                                                  GPUTarget::BIFROST,
                                                                                  GPUTarget::BIFROST,
                                                                                  GPUTarget::BIFROST,
                                                                                  GPUTarget::BIFROST,
                                          })),
                                          framework::dataset::make("Dilation", { Size2D(1U, 1U),
                                                                 Size2D(1U, 1U),
                                                                 Size2D(1U, 1U),
                                                                 Size2D(1U, 1U),
                                                                 Size2D(1U, 1U),
                                                                 Size2D(1U, 1U),
                                                                 Size2D(1U, 1U),
                                                                 Size2D(2U, 1U),
                                                                 Size2D(2U, 1U),
                                          })),
                                         framework::dataset::make("EnableFastMath", { false, false, false, false, false, false, true, true, true })),
                                         framework::dataset::make("Expected",{ ConvolutionMethod::GEMM,
                                                                               ConvolutionMethod::GEMM,
                                                                               ConvolutionMethod::GEMM,
                                                                               ConvolutionMethod::WINOGRAD,
                                                                               ConvolutionMethod::GEMM,
                                                                               ConvolutionMethod::GEMM,
                                                                               ConvolutionMethod::WINOGRAD,
                                                                               ConvolutionMethod::GEMM,
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

DATA_TEST_CASE(ValidatePostOpSupportInConvolutionMethod, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(
                                          framework::dataset::make("InputInfo", { TensorInfo(TensorShape(2U, 17U, 31U), 1, DataType::F32, DataLayout::NHWC),            // Select GEMM
                                                                                  TensorInfo(TensorShape(17U, 31U, 32U), 1, DataType::F32, DataLayout::NCHW),           // Select WINOGRAD
                                                                                  TensorInfo(TensorShape(27U, 27U, 48U), 1, DataType::F32, DataLayout::NCHW),           // Select Direct
                                                                                  TensorInfo(TensorShape(27U, 27U, 48U), 1, DataType::F32, DataLayout::NCHW),           // Select FFT
                                          }),
                                          framework::dataset::make("WeightsInfo", { TensorInfo(TensorShape(2U, 1U, 1U, 19U), 1, DataType::F32, DataLayout::NHWC),
                                                                                    TensorInfo(TensorShape(5U, 5U, 32U, 19U), 1, DataType::F32, DataLayout::NCHW),
                                                                                    TensorInfo(TensorShape(5U, 5U, 48U, 128U), 1, DataType::F32, DataLayout::NCHW),
                                                                                    TensorInfo(TensorShape(11U, 11U, 48U, 24), 1, DataType::F32, DataLayout::NCHW),
                                          })),
                                          framework::dataset::make("OutputInfo", { TensorInfo(TensorShape(19U, 17U, 31U), 1, DataType::F32, DataLayout::NHWC),
                                                                                   TensorInfo(TensorShape(17U, 31U, 19U), 1, DataType::F32, DataLayout::NCHW),
                                                                                   TensorInfo(TensorShape(27U, 27U, 128U), 1, DataType::F32, DataLayout::NCHW),
                                                                                   TensorInfo(TensorShape(27U, 27U, 24U), 1, DataType::F32, DataLayout::NCHW),
                                          })),
                                          framework::dataset::make("ConvInfo", { PadStrideInfo(1U, 1U, 0U, 0U),
                                                                                 PadStrideInfo(1U, 1U, 2U, 2U),
                                                                                 PadStrideInfo(1U, 1U, 2U, 2U),
                                                                                 PadStrideInfo(1U, 1U, 5U, 5U),
                                          })),
                                         framework::dataset::make("EnableFastMath", { false, true, false, false})),
                                         framework::dataset::make("ExpectedMethod",{ ConvolutionMethod::GEMM,
                                                                                     ConvolutionMethod::WINOGRAD,
                                                                                     ConvolutionMethod::DIRECT,
                                                                                     ConvolutionMethod::FFT,
                                         })),
                                         framework::dataset::make("PostOpSupported",{ true, false, false, false
                                         })),
                                         input_info, weights_info, output_info, conv_info, enable_fast_math, expected_method, post_op_supported)
{
    const int idx_width  = get_data_layout_dimension_index(input_info.data_layout(), DataLayoutDimension::WIDTH);
    const int idx_height = get_data_layout_dimension_index(input_info.data_layout(), DataLayoutDimension::HEIGHT);
    const int idx_kernels = get_data_layout_dimension_index(input_info.data_layout(), DataLayoutDimension::BATCHES);

    const auto dilation = Size2D(1U, 1U);
    const unsigned int num_groups = 1U;

    WeightsInfo w_info(false, weights_info.dimension(idx_width), weights_info.dimension(idx_height), weights_info.dimension(idx_kernels));

    experimental::PostOpList<ITensorInfo*> post_ops{};
    post_ops.push_back_op<experimental::PostOpAct<ITensorInfo*>>(ActivationLayerInfo{ActivationLayerInfo::ActivationFunction::LINEAR, 0.5F, 0.0F});

    ConvolutionMethod actual_method = CLConvolutionLayer::get_convolution_method(&input_info.clone()->set_is_resizable(true),
                                                                            &weights_info.clone()->set_is_resizable(true),
                                                                            &output_info.clone()->set_is_resizable(true), conv_info,
                                                                            WeightsInfo(),
                                                                            ActivationLayerInfo(),
                                                                            GPUTarget::BIFROST,
                                                                            dilation,
                                                                            enable_fast_math);
    ARM_COMPUTE_EXPECT(actual_method == expected_method, framework::LogLevel::ERRORS);
    const auto is_valid = CLConvolutionLayer::validate(&input_info.clone()->set_is_resizable(true),
                                                                            &weights_info.clone()->set_is_resizable(true),
                                                                            nullptr,
                                                                            &output_info.clone()->set_is_resizable(true),
                                                                            conv_info,
                                                                            w_info,
                                                                            dilation,
                                                                            ActivationLayerInfo(),
                                                                            enable_fast_math,
                                                                            num_groups,
                                                                            post_ops);
    ARM_COMPUTE_EXPECT( bool(is_valid) == post_op_supported, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*
TEST_SUITE_END() // ConvolutionLayer

TEST_SUITE(GEMMConvolutionLayer)
template <typename T>
using CLGEMMConvolutionLayerFixture = ConvolutionValidationFixture<CLTensor, CLAccessor, CLGEMMConvolutionLayer, T>;
template <typename T>
using CLGEMMConvolutionLayerMixedDataLayoutFixture = ConvolutionValidationFixture<CLTensor, CLAccessor, CLGEMMConvolutionLayer, T, true>;
template <typename T>
using CLConvolutionValidationWithPaddingFixture = ConvolutionValidationWithPaddingFixture<CLTensor, CLAccessor, CLGEMMConvolutionLayer, T>;

TEST_SUITE(ValidateFusedPostOpsConfigs)
TEST_SUITE(Invalid)
TEST_CASE(UnsupportedPostOpSequence, framework::DatasetMode::ALL)
{
    const auto data_type     = DataType::F32;
    const auto data_layout   = DataLayout::NHWC;
    const auto conv_info     = PadStrideInfo(1, 1, 0, 0);
    const auto input_shape   = TensorShape(16U, 14U, 12U, 2U);
    const auto weights_shape = TensorShape(16U, 1U, 1U, 24U);

    const auto output_shape = misc::shape_calculator::compute_deep_convolution_shape(input_shape, data_layout, weights_shape, conv_info);

    const TensorShape post_op_arg0_shape(output_shape);
    TensorInfo        post_op_arg_info(post_op_arg0_shape, 1, data_type);
    auto              post_op_arg1_info = post_op_arg_info.clone();

    // Unsupported sequence of post ops
    experimental::PostOpList<ITensorInfo *> post_ops{};
    post_ops.push_back_op<experimental::PostOpEltwiseAdd<ITensorInfo *>>(
                                                                          &post_op_arg_info,
                                                                          1,
                                                                          ConvertPolicy::SATURATE);
    post_ops.push_back_op<experimental::PostOpEltwiseAdd<ITensorInfo *>>(
                                                                          post_op_arg1_info.get(),
                                                                          0,
                                                                          ConvertPolicy::SATURATE);

    ARM_COMPUTE_EXPECT(is_post_op_list_valid_in_gemmconv(input_shape, weights_shape, output_shape, data_type, data_layout, conv_info, post_ops) == false, framework::LogLevel::ERRORS);
}
TEST_CASE(OnlyNHWCIsSupported, framework::DatasetMode::ALL)
{
    const auto data_type     = DataType::F32;
    const auto data_layout   = DataLayout::NCHW;
    const auto conv_info     = PadStrideInfo(1, 1, 0, 0);
    const auto input_shape   = TensorShape(14U, 12U, 16U, 2U);
    const auto weights_shape = TensorShape(1U, 1U, 16U, 24U);

    const auto output_shape = misc::shape_calculator::compute_deep_convolution_shape(input_shape, data_layout, weights_shape, conv_info);

    const TensorShape post_op_arg0_shape(output_shape);
    TensorInfo        post_op_arg_info(post_op_arg0_shape, 1, data_type);

    experimental::PostOpList<ITensorInfo *> post_ops{};
    post_ops.push_back_op<experimental::PostOpEltwiseAdd<ITensorInfo *>>(
                                                                          &post_op_arg_info,
                                                                          1,
                                                                          ConvertPolicy::SATURATE);

    ARM_COMPUTE_EXPECT(is_post_op_list_valid_in_gemmconv(input_shape, weights_shape, output_shape, data_type, data_layout, conv_info, post_ops) == false, framework::LogLevel::ERRORS);
}
TEST_CASE(OnlyFloatingTypeIsSupported, framework::DatasetMode::ALL)
{
    const auto data_type     = DataType::QASYMM8;
    const auto data_layout   = DataLayout::NHWC;
    const auto conv_info     = PadStrideInfo(1, 1, 0, 0);
    const auto input_shape   = TensorShape(16U, 14U, 12U, 2U);
    const auto weights_shape = TensorShape(16U, 1U, 1U, 24U);

    const auto output_shape = misc::shape_calculator::compute_deep_convolution_shape(input_shape, data_layout, weights_shape, conv_info);

    const TensorShape post_op_arg0_shape(output_shape);
    TensorInfo        post_op_arg_info(post_op_arg0_shape, 1, data_type);

    experimental::PostOpList<ITensorInfo *> post_ops{};
    post_ops.push_back_op<experimental::PostOpEltwiseAdd<ITensorInfo *>>(
                                                                          &post_op_arg_info,
                                                                          1,
                                                                          ConvertPolicy::SATURATE);

    ARM_COMPUTE_EXPECT(is_post_op_list_valid_in_gemmconv(input_shape, weights_shape, output_shape, data_type, data_layout, conv_info, post_ops) == false, framework::LogLevel::ERRORS);
}
TEST_CASE(OnlyConv1x1Stride1IsSupported_UnsupportedKernelSize, framework::DatasetMode::ALL)
{
    const auto data_type     = DataType::F32;
    const auto data_layout   = DataLayout::NHWC;
    const auto conv_info     = PadStrideInfo(1, 1, 0, 0);
    const auto input_shape   = TensorShape(16U, 14U, 12U, 2U);
    const auto weights_shape = TensorShape(16U, 3U, 3U, 24U);

    const auto output_shape = misc::shape_calculator::compute_deep_convolution_shape(input_shape, data_layout, weights_shape, conv_info);

    const TensorShape post_op_arg0_shape(output_shape);
    TensorInfo        post_op_arg_info(post_op_arg0_shape, 1, data_type);

    experimental::PostOpList<ITensorInfo *> post_ops{};
    post_ops.push_back_op<experimental::PostOpEltwiseAdd<ITensorInfo *>>(
                                                                          &post_op_arg_info,
                                                                          1,
                                                                          ConvertPolicy::SATURATE);

    ARM_COMPUTE_EXPECT(is_post_op_list_valid_in_gemmconv(input_shape, weights_shape, output_shape, data_type, data_layout, conv_info, post_ops) == false, framework::LogLevel::ERRORS);
}
TEST_CASE(OnlyConv1x1Stride1IsSupported_UnsupportedStride, framework::DatasetMode::ALL)
{
    const auto data_type     = DataType::F32;
    const auto data_layout   = DataLayout::NHWC;
    const auto conv_info     = PadStrideInfo(3, 3, 0, 0);
    const auto input_shape   = TensorShape(16U, 14U, 12U, 2U);
    const auto weights_shape = TensorShape(16U, 1U, 1U, 24U);

    const auto output_shape = misc::shape_calculator::compute_deep_convolution_shape(input_shape, data_layout, weights_shape, conv_info);

    const TensorShape post_op_arg0_shape(output_shape);
    TensorInfo        post_op_arg_info(post_op_arg0_shape, 1, data_type);

    experimental::PostOpList<ITensorInfo *> post_ops{};
    post_ops.push_back_op<experimental::PostOpEltwiseAdd<ITensorInfo *>>(
                                                                          &post_op_arg_info,
                                                                          1,
                                                                          ConvertPolicy::SATURATE);

    ARM_COMPUTE_EXPECT(is_post_op_list_valid_in_gemmconv(input_shape, weights_shape, output_shape, data_type, data_layout, conv_info, post_ops) == false, framework::LogLevel::ERRORS);
}
TEST_SUITE_END() // Invalid
TEST_SUITE(Valid)
TEST_CASE(EmptyPostOpList, framework::DatasetMode::ALL)
{
    const auto data_type     = DataType::F32;
    const auto data_layout   = DataLayout::NHWC;
    const auto conv_info     = PadStrideInfo(1, 1, 0, 0);
    const auto input_shape   = TensorShape(16U, 14U, 12U, 2U);
    const auto weights_shape = TensorShape(16U, 1U, 1U, 24U);

    const auto output_shape = misc::shape_calculator::compute_deep_convolution_shape(input_shape, data_layout, weights_shape, conv_info);

    experimental::PostOpList<ITensorInfo *> post_ops{};

    ARM_COMPUTE_EXPECT(is_post_op_list_valid_in_gemmconv(input_shape, weights_shape, output_shape, data_type, data_layout, conv_info, post_ops) == true, framework::LogLevel::ERRORS);
}
TEST_CASE(SupportedPostOps, framework::DatasetMode::ALL)
{
    const auto data_type     = DataType::F32;
    const auto data_layout   = DataLayout::NHWC;
    const auto conv_info     = PadStrideInfo(1, 1, 0, 0);
    const auto input_shape   = TensorShape(16U, 14U, 12U, 2U);
    const auto weights_shape = TensorShape(16U, 1U, 1U, 24U);

    const auto output_shape = misc::shape_calculator::compute_deep_convolution_shape(input_shape, data_layout, weights_shape, conv_info);

    TensorShape post_op_arg0_shape(output_shape);
    post_op_arg0_shape[1] = 1; // Broadcast in "Y" (second) dimension
    TensorInfo post_op_arg_info(post_op_arg0_shape, 1, data_type);

    experimental::PostOpList<ITensorInfo *> post_ops{};
    post_ops.push_back_op<experimental::PostOpEltwiseAdd<ITensorInfo *>>(
                                                                          &post_op_arg_info,
                                                                          1,
                                                                          ConvertPolicy::SATURATE);

    ARM_COMPUTE_EXPECT(is_post_op_list_valid_in_gemmconv(input_shape, weights_shape, output_shape, data_type, data_layout, conv_info, post_ops) == true, framework::LogLevel::ERRORS);
}
TEST_SUITE_END() // Valid
TEST_SUITE_END() // ValidateFusedPostOps
TEST_SUITE(Float)
TEST_SUITE(FP16)

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMConvolutionLayerFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                                                                   framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                   framework::dataset::make("DataType",
                                                                                                                           DataType::F16)),
                                                                                                                   framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                           ActivationFunctionsSmallDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
TEST_SUITE_END() // FP16

TEST_SUITE(FP32)

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMConvolutionLayerFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                                                                    framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                    framework::dataset::make("DataType",
                                                                                                                            DataType::F32)),
                                                                                                                    framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                                                                                            ActivationFunctionsSmallDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunMixedDataLayout, CLGEMMConvolutionLayerMixedDataLayoutFixture<float>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                                           framework::dataset::make("Input", TensorShape(23U, 27U, 5U)),
                                                                                           framework::dataset::make("Weights", TensorShape(3U, 3U, 5U, 2U))),
                                                                                       framework::dataset::make("Bias", TensorShape(2U))),
                                                                               framework::dataset::make("Output", TensorShape(11U, 25U, 2U))),
                                                                       framework::dataset::make("PadStrideInfo", PadStrideInfo(2, 1, 0, 0))),
                                                               framework::dataset::make("Dilation", Size2D(1, 1))),
                                                       framework::dataset::make("ReshapeWeights", { true })),
                                               framework::dataset::make("DataType", DataType::F32)),
                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                               ActivationFunctionsSmallDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunSmallWithPadding, CLConvolutionValidationWithPaddingFixture<float>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(datasets::SmallConvolutionLayerPrePaddingDataset(),
                                                               framework::dataset::make("ReshapeWeights", { true })),
                                                       framework::dataset::make("DataType", DataType::F32)),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                       framework::dataset::make("ActivationInfo", { ActivationLayerInfo() })),
framework::dataset::make("PrePadLayer", { PaddingList({ { 1, 1 }, { 1, 1 } }) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

template <typename T>
using CLGEMMConvolutionLayerQuantizedFixture = ConvolutionValidationQuantizedFixture<CLTensor, CLAccessor, CLGEMMConvolutionLayer, T>;
template <typename T>
using CLGEMMConvolutionLayerQuantizedMixedDataLayoutFixture = ConvolutionValidationQuantizedFixture<CLTensor, CLAccessor, CLGEMMConvolutionLayer, T, true>;
template <typename T>
using CLGEMMConvolutionLayerQuantizedPerChannelFixture = ConvolutionValidationQuantizedPerChannelFixture<CLTensor, CLAccessor, CLGEMMConvolutionLayer, T, int8_t>;

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

const auto QuantizationData = framework::dataset::make("QuantizationInfo",
{
    QuantizationInfo(0.5f, 10),
    QuantizationInfo(0.3f, 3),
    QuantizationInfo(1.1f, 10),
});
TEST_SUITE(QASYMM8)

FIXTURE_DATA_TEST_CASE(RunSmallCases, CLGEMMConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(SmallConvolutionLayerDatasetCases(),
                                                               framework::dataset::make("ReshapeWeights", { true })),
                                                       framework::dataset::make("DataType", DataType::QASYMM8)),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                       QuantizationData),
                               QuantizedActivationFunctionsSmallDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMConvolutionLayerQuantizedFixture<uint8_t>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                               framework::dataset::make("ReshapeWeights", { true })),
                                                       framework::dataset::make("DataType", DataType::QASYMM8)),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                       QuantizationData),
                               QuantizedActivationFunctionsSmallDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunMixedDataLayout, CLGEMMConvolutionLayerQuantizedMixedDataLayoutFixture<uint8_t>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                                                   framework::dataset::make("Input", TensorShape(23U, 27U, 5U)),
                                                                                                   framework::dataset::make("Weights", TensorShape(3U, 3U, 5U, 2U))),
                                                                                               framework::dataset::make("Bias", TensorShape(2U))),
                                                                                       framework::dataset::make("Output", TensorShape(11U, 25U, 2U))),
                                                                               framework::dataset::make("PadStrideInfo", PadStrideInfo(2, 1, 0, 0))),
                                                                       framework::dataset::make("Dilation", Size2D(1, 1))),
                                                               framework::dataset::make("ReshapeWeights", { true })),
                                                       framework::dataset::make("DataType", DataType::QASYMM8)),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                       QuantizationData),
                               QuantizedActivationFunctionsSmallDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMConvolutionLayerQuantizedFixture<int8_t>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                               framework::dataset::make("ReshapeWeights", { true })),
                                                       framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                       QuantizationData),
                               QuantizedActivationFunctionsSmallDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunMixedDataLayout, CLGEMMConvolutionLayerQuantizedMixedDataLayoutFixture<int8_t>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(combine(combine(combine(combine(
                                                                                                   framework::dataset::make("Input", TensorShape(23U, 27U, 5U)),
                                                                                                   framework::dataset::make("Weights", TensorShape(3U, 3U, 5U, 2U))),
                                                                                               framework::dataset::make("Bias", TensorShape(2U))),
                                                                                       framework::dataset::make("Output", TensorShape(11U, 25U, 2U))),
                                                                               framework::dataset::make("PadStrideInfo", PadStrideInfo(2, 1, 0, 0))),
                                                                       framework::dataset::make("Dilation", Size2D(1, 1))),
                                                               framework::dataset::make("ReshapeWeights", { true })),
                                                       framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                                               framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                       QuantizationData),
                               QuantizedActivationFunctionsSmallDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE(QSYMM8_PER_CHANNEL)

FIXTURE_DATA_TEST_CASE(RunSmallSigned, CLGEMMConvolutionLayerQuantizedPerChannelFixture<int8_t>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                       framework::dataset::make("ReshapeWeights", { true })),
                                                               framework::dataset::make("DataType", { DataType::QASYMM8_SIGNED })),
                                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                               QuantizationData),
                                       QuantizedActivationFunctionsSmallDataset),
                               framework::dataset::make("WeightsDataType", { DataType::QSYMM8_PER_CHANNEL })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMConvolutionLayerQuantizedPerChannelFixture<uint8_t>, framework::DatasetMode::ALL,
                       combine(combine(combine(combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                       framework::dataset::make("ReshapeWeights", { true })),
                                                               framework::dataset::make("DataType", { DataType::QASYMM8 })),
                                                       framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })),
                                               QuantizationData),
                                       QuantizedActivationFunctionsSmallDataset),
                               framework::dataset::make("WeightsDataType", { DataType::QSYMM8_PER_CHANNEL })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QSYMM8_PER_CHANNEL
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // GEMMConvolutionLayer

template <typename T>
using CLGEMMGroupedConvolutionLayerFixture = ConvolutionValidationFixture<CLTensor, CLAccessor, CLGEMMConvolutionLayer, T>;

TEST_SUITE(GroupedGEMMConvolutionLayer)

TEST_SUITE(Float)
TEST_SUITE(FP32)

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMGroupedConvolutionLayerFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallGroupedConvolutionLayerDataset(),
                                                                                                                   framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                                                                   framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                                                                                                   ActivationFunctionsSmallDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32, tolerance_num);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMGroupedConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(datasets::LargeGroupedConvolutionLayerDataset(),
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

FIXTURE_DATA_TEST_CASE(RunSmall, CLGEMMGroupedConvolutionLayerFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(combine(datasets::SmallGroupedConvolutionLayerDataset(),
                                                                                                                  framework::dataset::make("ReshapeWeights", { true })),
                                                                                                                  framework::dataset::make("DataType", DataType::F16)),
                                                                                                                  framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                                                                                                                  ActivationFunctionsSmallDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32, tolerance_num);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLGEMMGroupedConvolutionLayerFixture<half>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(combine(datasets::LargeGroupedConvolutionLayerDataset(),
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
