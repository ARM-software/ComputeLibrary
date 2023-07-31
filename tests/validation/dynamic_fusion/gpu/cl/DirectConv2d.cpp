/*
 * Copyright (c) 2022-2023 Arm Limited.
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

#include "tests/AssetsLibrary.h"
#include "tests/CL/CLAccessor.h"
#include "tests/framework/Fixture.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/reference/ConvolutionLayer.h"

#include "tests/datasets/SmallConvolutionLayerDataset.h"
#include "tests/validation/fixtures/dynamic_fusion/gpu/cl/DirectConv2dFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Tolerances from tests/validation/CL/DirectConvolutionLayer.cpp
 */
RelativeTolerance<float>            tolerance_f32(0.05f);                 /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
RelativeTolerance<half_float::half> tolerance_f16(half_float::half(0.2)); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F16 */
constexpr float                     abs_tolerance_f32(0.0001f);           /**< Absolute tolerance for FP32 tests*/
constexpr float                     tolerance_num = 0.07f;                /**< Tolerance number */
} // namespace

TEST_SUITE(CL)
TEST_SUITE(DYNAMIC_FUSION)
/** Synced with tests/validation/CL/ConvolutionLayer.cpp
 *
 * Difference                       | Why the difference
 * f32 tolerance here is smaller    | To use the same tolerance as that of DirectConv2d; lowering tolerance is safe
 * No quantized tests               | Not supported yet
 * No grouped CNN tests             | Not supported yet
 * No mixed layout tests            | Not needed; only NHWC is supported
 * No activation/post op tests      | Not needed in fusion
 * No ValidateConvolutionMethod     | Only a single method (direct conv2d) is supported
 * No ReshapeWeights = true tests   | Not applicable yet. This parameter only concerns gemm-based conv2d
 * No RunSmallWithPadding tests     | Padding is removed
 *
 */
TEST_SUITE(CONV2D)

template <typename T>
using DynamicFusionGpuConv2dFixture = DynamicFusionGpuConv2dValidationFixture<CLTensor, CLAccessor, GpuConv2d, T>;
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, DynamicFusionGpuConv2dFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                                                                    framework::dataset::make("DataType", DataType::F32)),
                                                                                                                    framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                                                                                            framework::dataset::make("QuantizationInfo", QuantizationInfo())))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // FP32

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, DynamicFusionGpuConv2dFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(datasets::SmallConvolutionLayerDataset(),
                                                                                                                   framework::dataset::make("DataType", DataType::F16)),
                                                                                                                   framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                                                                                           framework::dataset::make("QuantizationInfo", QuantizationInfo())))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
TEST_SUITE_END() // FP16

// Tests for specific conv2d methods
/** Synced with tests/validation/CL/DirectConvolutionLayer.cpp
 *
 * Difference                       | Why the difference
 * No quantized tests               | Not supported yet
 * No Invalid output size test      | Not applicable. Output is removed from the interface
 * No mixed layout/NCHW tests       | Not needed; only NHWC is supported
 * No activation tests              | Not needed in fusion
 */
TEST_SUITE(DIRECT_CONV2D)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(2U, 27U, 13U), 1, DataType::F32, DataLayout::NHWC),       // Invalid: Mismatching data type input/weights
                                                       TensorInfo(TensorShape(2U, 27U, 13U), 1, DataType::F32, DataLayout::NHWC),       // Invalid: Mismatching input feature maps
                                                       TensorInfo(TensorShape(2U, 27U, 13U), 1, DataType::F32, DataLayout::NHWC),       // Invalid weights dimensions
                                                       TensorInfo(TensorShape(2U, 27U, 13U), 1, DataType::F32, DataLayout::NHWC),       // Unsupported biases size
                                                       TensorInfo(TensorShape(2U, 27U, 13U), 1, DataType::F32, DataLayout::NHWC),       // Unsupported biases dimensions
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32, DataLayout::NCHW),       // Unsupported data layout: NCHW
                                                       TensorInfo(TensorShape(2U, 32U, 16U), 1, DataType::QASYMM8, DataLayout::NHWC),   // Unsupported data type: quantized
                                                       TensorInfo(TensorShape(2U, 32U, 16U), 1, DataType::F32, DataLayout::NHWC),
                                                       TensorInfo(TensorShape(2U, 27U, 13U), 1, DataType::F32, DataLayout::NHWC),       // Arbitrary weight sizes for NHWC are supported
                                                       TensorInfo(TensorShape(2U, 27U, 13U), 1, DataType::F32, DataLayout::NHWC),       // Non-rectangular weights dimensions for NHWC are supported
                                                       TensorInfo(TensorShape(2U, 27U, 13U), 1, DataType::F32, DataLayout::NHWC),       // Strides > 2 for any kernel sizes for NHWC are supported
                                                     }),
               framework::dataset::make("WeightsInfo",{ TensorInfo(TensorShape(2U, 3U, 3U, 4U), 1, DataType::F16, DataLayout::NHWC),
                                                        TensorInfo(TensorShape(3U, 3U, 3U, 4U), 1, DataType::F32, DataLayout::NHWC),
                                                        TensorInfo(TensorShape(2U, 3U, 3U, 4U, 3U), 1, DataType::F32, DataLayout::NHWC),
                                                        TensorInfo(TensorShape(2U, 3U, 3U, 4U), 1, DataType::F32, DataLayout::NHWC),
                                                        TensorInfo(TensorShape(2U, 3U, 3U, 4U), 1, DataType::F32, DataLayout::NHWC),
                                                        TensorInfo(TensorShape(3U, 3U, 2U, 4U), 1, DataType::F32, DataLayout::NCHW),
                                                        TensorInfo(TensorShape(2U, 1U, 1U, 4U), 1, DataType::QASYMM8, DataLayout::NHWC),
                                                        TensorInfo(TensorShape(2U, 1U, 1U, 4U), 1, DataType::F32, DataLayout::NHWC),
                                                        TensorInfo(TensorShape(2U, 13U, 13U, 4U), 1, DataType::F32, DataLayout::NHWC),
                                                        TensorInfo(TensorShape(2U, 5U, 3U, 4U), 1, DataType::F32, DataLayout::NHWC),
                                                        TensorInfo(TensorShape(2U, 3U, 3U, 4U), 1, DataType::F32, DataLayout::NHWC),
                                                     })),
               framework::dataset::make("BiasesInfo",{ TensorInfo(TensorShape(4U), 1, DataType::F32, DataLayout::NHWC),
                                                       TensorInfo(TensorShape(4U), 1, DataType::F32, DataLayout::NHWC),
                                                       TensorInfo(TensorShape(4U), 1, DataType::F32, DataLayout::NHWC),
                                                       TensorInfo(TensorShape(3U), 1, DataType::F32, DataLayout::NHWC),
                                                       TensorInfo(TensorShape(4U, 2U), 1, DataType::F32, DataLayout::NHWC),
                                                       TensorInfo(TensorShape(25U), 1, DataType::F32, DataLayout::NCHW),
                                                       TensorInfo(TensorShape(4U), 1, DataType::QASYMM8, DataLayout::NHWC),
                                                       TensorInfo(TensorShape(4U), 1, DataType::F32, DataLayout::NHWC),
                                                       TensorInfo(TensorShape(4U), 1, DataType::F32, DataLayout::NHWC),
                                                       TensorInfo(TensorShape(4U), 1, DataType::F32, DataLayout::NHWC),
                                                       TensorInfo(TensorShape(4U), 1, DataType::F32, DataLayout::NHWC),
                                                     })),
               framework::dataset::make("Conv2dAttributes",  {
                                                        Conv2dAttributes().stride({1, 1}).pad({0, 0, 0, 0}),
                                                        Conv2dAttributes().stride({1, 1}).pad({0, 0, 0, 0}),
                                                        Conv2dAttributes().stride({1, 1}).pad({0, 0, 0, 0}),
                                                        Conv2dAttributes().stride({1, 1}).pad({0, 0, 0, 0}),
                                                        Conv2dAttributes().stride({1, 1}).pad({0, 0, 0, 0}),
                                                        Conv2dAttributes().stride({1, 1}).pad({0, 0, 0, 0}),
                                                        Conv2dAttributes().stride({1, 1}).pad({0, 0, 0, 0}),
                                                        Conv2dAttributes().stride({1, 1}).pad({0, 0, 0, 0}),
                                                        Conv2dAttributes().stride({1, 1}).pad({0, 0, 0, 0}),
                                                        Conv2dAttributes().stride({1, 1}).pad({0, 0, 0, 0}),
                                                        Conv2dAttributes().stride({3, 3}).pad({0, 0, 0, 0}),
                                                      })),
               framework::dataset::make("Expected", { false, false, false, false, false, false, false, true, true, true, true })),
               input_info, weights_info, biases_info, conv2d_attrs, expected)
{
    auto              cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
    auto              context        = GpuWorkloadContext{ &cl_compile_ctx };
    GpuWorkloadSketch sketch{ &context };

    const TensorInfo sketch_input_info   = context.create_tensor_info(input_info);
    const TensorInfo sketch_weights_info = context.create_tensor_info(weights_info);
    const TensorInfo sketch_biases_info  = context.create_tensor_info(biases_info);
    bool is_valid = bool(GpuConv2d::validate_op(sketch, &sketch_input_info, &sketch_weights_info, &sketch_biases_info, conv2d_attrs));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
template <typename T>
using DynamicFusionGpuDirectConv2dFixture = DynamicFusionDirectConv2dValidationFixture<CLTensor, CLAccessor, GpuConv2d, T>;

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, DynamicFusionGpuDirectConv2dFixture<half>, framework::DatasetMode::PRECOMMIT,
               combine(combine(combine(zip(zip(zip(zip(zip(
               framework::dataset::make("InputShape", { TensorShape(27U, 13U, 23U),
                                                        TensorShape(19U, 5U, 16U, 4U),
                                                        TensorShape(13U, 5U, 17U, 2U),
                                                        TensorShape(32U, 37U, 13U) } ),
               framework::dataset::make("StrideX", { 1, 3, 1, 1 })),
               framework::dataset::make("StrideY", { 1, 3, 2, 1 })),
               framework::dataset::make("PadX", { 1, 3, 0, 4 })),
               framework::dataset::make("PadY", { 1, 3, 0, 4 })),
               framework::dataset::make("KernelSize", { 3, 8, 1, 9 })),
               framework::dataset::make("NumKernels", { 17, 3, 1, 19 })),
               framework::dataset::make("DataType",  DataType::F16)),
               framework::dataset::make("DataLayout", DataLayout::NHWC)))
{
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}

FIXTURE_DATA_TEST_CASE(RunLarge, DynamicFusionGpuDirectConv2dFixture<half>, framework::DatasetMode::NIGHTLY,
               combine(combine(combine(zip(zip(zip(zip(zip(
               framework::dataset::make("InputShape", { TensorShape(800U, 800U, 3U) } ),
               framework::dataset::make("StrideX", { 1 })),
               framework::dataset::make("StrideY", { 1 })),
               framework::dataset::make("PadX", { 1 })),
               framework::dataset::make("PadY", { 1 })),
               framework::dataset::make("KernelSize", { 9 })),
               framework::dataset::make("NumKernels", { 3 })),
               framework::dataset::make("DataType",  DataType::F16)),
               framework::dataset::make("DataLayout", DataLayout::NHWC)))
{
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}

TEST_SUITE_END() // FP16

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, DynamicFusionGpuDirectConv2dFixture<float>, framework::DatasetMode::PRECOMMIT,
               combine(combine(combine(zip(zip(zip(zip(zip(
               framework::dataset::make("InputShape", { TensorShape(27U, 13U, 23U),
                                                        TensorShape(19U, 5U, 16U, 4U),
                                                        TensorShape(13U, 5U, 17U, 2U),
                                                        TensorShape(32U, 37U, 13U) } ),
               framework::dataset::make("StrideX", { 1, 3, 1, 1 })),
               framework::dataset::make("StrideY", { 1, 3, 2, 1 })),
               framework::dataset::make("PadX", { 1, 3, 0, 4 })),
               framework::dataset::make("PadY", { 1, 3, 0, 4 })),
               framework::dataset::make("KernelSize", { 3, 8, 1, 9 })),
               framework::dataset::make("NumKernels", { 17, 3, 1, 19 })),
               framework::dataset::make("DataType",  DataType::F32)),
               framework::dataset::make("DataLayout", DataLayout::NHWC)))
{
    validate(CLAccessor(_target), _reference, tolerance_f32, 0.0, abs_tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, DynamicFusionGpuDirectConv2dFixture<float>, framework::DatasetMode::NIGHTLY,
               combine(combine(combine(zip(zip(zip(zip(zip(
               framework::dataset::make("InputShape", { TensorShape(800U, 800U, 3U) } ),
               framework::dataset::make("StrideX", { 1 })),
               framework::dataset::make("StrideY", { 1 })),
               framework::dataset::make("PadX", { 1 })),
               framework::dataset::make("PadY", { 1 })),
               framework::dataset::make("KernelSize", { 9 })),
               framework::dataset::make("NumKernels", { 3 })),
               framework::dataset::make("DataType",  DataType::F32)),
               framework::dataset::make("DataLayout", DataLayout::NHWC)))
{
    validate(CLAccessor(_target), _reference, tolerance_f32, 0.0, abs_tolerance_f32);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE_END() // FP32
TEST_SUITE_END() // DIRECT_CONV2D
TEST_SUITE_END() // CONV2D
TEST_SUITE_END() // DYNAMIC_FUSION
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
