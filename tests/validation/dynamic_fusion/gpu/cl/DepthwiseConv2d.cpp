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

#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuDepthwiseConv2d.h"

#include "tests/CL/CLAccessor.h"
#include "tests/datasets/DepthwiseConvolutionLayerDataset.h"
#include "tests/datasets/DilatedDepthwiseConvolutionLayerDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/dynamic_fusion/gpu/cl/DepthwiseConv2dFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
const auto depth_multipliers       = framework::dataset::make("DepthMultiplier", { 1U, 4U });
const auto large_depth_multipliers = framework::dataset::make("DepthMultiplier", { 1, 2, 5, 8 });

TEST_SUITE(CL)
TEST_SUITE(DYNAMIC_FUSION)
TEST_SUITE(DEPTHWISE_CONV2D)

RelativeTolerance<float>            tolerance_f32(0.01f);                 /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
RelativeTolerance<half_float::half> tolerance_f16(half_float::half(0.1)); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F16 */
constexpr float                     tolerance_num = 0.02f;                /**< Tolerance number */

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(zip(                                                                  // Explanations of failing tests
                framework::dataset::make("InputInfo", { TensorInfo(TensorShape(2U, 27U, 13U), 1, DataType::F32, DataLayout::NHWC),                  // Mismatching data type input/weights
                                                        TensorInfo(TensorShape(3U, 27U, 13U), 1, DataType::F32, DataLayout::NHWC),                  // Mismatching input feature maps
                                                        TensorInfo(TensorShape(2U, 27U, 13U), 1, DataType::F32, DataLayout::NHWC),                  // Mismatching depth multiplier
                                                        TensorInfo(TensorShape(2U, 27U, 13U), 1, DataType::F32, DataLayout::NHWC),                  // Invalid biases size
                                                        TensorInfo(TensorShape(2U, 27U, 13U), 1, DataType::F32, DataLayout::NHWC),                  // Invalid biases dimensions
                                                        TensorInfo(TensorShape(8U, 27U, 13U), 1, DataType::F32, DataLayout::NHWC),                  // dilation < 1
                                                        TensorInfo(TensorShape(8U, 27U, 13U), 1, DataType::F32, DataLayout::NHWC),
                                                        TensorInfo(TensorShape(8U, 32U, 13U), 1, DataType::QASYMM8, DataLayout::NHWC),              // Unsupported data type
                                                        TensorInfo(TensorShape(8U, 32U, 13U), 1, DataType::QASYMM8_SIGNED, DataLayout::NHWC),       // Unsupported data type
                                                        TensorInfo(TensorShape(8U, 32U, 13U), 1, DataType::QSYMM16, DataLayout::NHWC),              // Unsupported data type
                                                        TensorInfo(TensorShape(8U, 32U, 13U), 1, DataType::QSYMM8, DataLayout::NHWC),               // Unsupported data type
                                                        TensorInfo(TensorShape(8U, 32U, 13U), 1, DataType::QSYMM8_PER_CHANNEL, DataLayout::NHWC),   // Unsupported data type
                                                        TensorInfo(TensorShape(8U, 32U, 13U), 1, DataType::QASYMM16, DataLayout::NHWC),             // Unsupported data type
                                                        TensorInfo(TensorShape(8U, 32U, 13U), 1, DataType::U8, DataLayout::NHWC),                   // Unsupported data type
                                                        TensorInfo(TensorShape(8U, 32U, 13U), 1, DataType::S8, DataLayout::NHWC),                   // Unsupported data type
                                                        TensorInfo(TensorShape(8U, 32U, 13U), 1, DataType::U16, DataLayout::NHWC),                  // Unsupported data type
                                                        TensorInfo(TensorShape(8U, 32U, 13U), 1, DataType::S16, DataLayout::NHWC),                  // Unsupported data type
                                                        TensorInfo(TensorShape(8U, 32U, 13U), 1, DataType::U32, DataLayout::NHWC),                  // Unsupported data type
                                                        TensorInfo(TensorShape(8U, 32U, 13U), 1, DataType::S32, DataLayout::NHWC),                  // Unsupported data type
                                                        TensorInfo(TensorShape(32U, 13U, 8U), 1, DataType::F32, DataLayout::NCHW),                  // Unsupported data layout
                                                        TensorInfo(TensorShape(8U, 32U, 13U, 4U), 1, DataType::F32, DataLayout::NHWC),
                                                        TensorInfo(TensorShape(8U, 32U, 13U, 4U), 1, DataType::F32, DataLayout::NHWC),              // weight dimension > 3
                                                        TensorInfo(TensorShape(8U, 32U, 13U, 4U), 1, DataType::F32, DataLayout::NHWC),
                                                        TensorInfo(TensorShape(8U, 32U, 13U, 4U), 1, DataType::F32, DataLayout::NHWC),
                                                        TensorInfo(TensorShape(8U, 32U, 13U, 4U), 1, DataType::F32, DataLayout::NHWC),
                                                      }),
                framework::dataset::make("WeightsInfo", { TensorInfo(TensorShape(2U, 3U, 3U, 2U), 1, DataType::F16, DataLayout::NHWC),
                                                          TensorInfo(TensorShape(2U, 3U, 3U, 2U), 1, DataType::F32, DataLayout::NHWC),
                                                          TensorInfo(TensorShape(2U, 3U, 3U, 2U), 1, DataType::F32, DataLayout::NHWC),
                                                          TensorInfo(TensorShape(2U, 3U, 3U, 2U), 1, DataType::F32, DataLayout::NHWC),
                                                          TensorInfo(TensorShape(2U, 3U, 3U, 2U), 1, DataType::F32, DataLayout::NHWC),
                                                          TensorInfo(TensorShape(16U, 3U, 3U), 1, DataType::F32, DataLayout::NHWC),
                                                          TensorInfo(TensorShape(16U, 3U, 3U), 1, DataType::F32, DataLayout::NHWC),
                                                          TensorInfo(TensorShape(24U, 3U, 3U), 1, DataType::QASYMM8, DataLayout::NHWC),
                                                          TensorInfo(TensorShape(24U, 3U, 3U), 1, DataType::QASYMM8_SIGNED, DataLayout::NHWC),
                                                          TensorInfo(TensorShape(24U, 3U, 3U), 1, DataType::QSYMM16, DataLayout::NHWC),
                                                          TensorInfo(TensorShape(24U, 3U, 3U), 1, DataType::QSYMM8, DataLayout::NHWC),
                                                          TensorInfo(TensorShape(24U, 3U, 3U), 1, DataType::QSYMM8_PER_CHANNEL, DataLayout::NHWC),
                                                          TensorInfo(TensorShape(24U, 3U, 3U), 1, DataType::QASYMM16, DataLayout::NHWC),
                                                          TensorInfo(TensorShape(24U, 3U, 3U), 1, DataType::U8, DataLayout::NHWC),
                                                          TensorInfo(TensorShape(24U, 3U, 3U), 1, DataType::S8, DataLayout::NHWC),
                                                          TensorInfo(TensorShape(24U, 3U, 3U), 1, DataType::U16, DataLayout::NHWC),
                                                          TensorInfo(TensorShape(24U, 3U, 3U), 1, DataType::S16, DataLayout::NHWC),
                                                          TensorInfo(TensorShape(24U, 3U, 3U), 1, DataType::U32, DataLayout::NHWC),
                                                          TensorInfo(TensorShape(24U, 3U, 3U), 1, DataType::S32, DataLayout::NHWC),
                                                          TensorInfo(TensorShape(3U, 3U, 24U), 1, DataType::F32, DataLayout::NCHW),
                                                          TensorInfo(TensorShape(24U, 3U, 3U), 1, DataType::F32, DataLayout::NHWC),
                                                          TensorInfo(TensorShape(24U, 3U, 3U, 5U), 1, DataType::F32, DataLayout::NHWC),
                                                          TensorInfo(TensorShape(24U, 3U, 3U), 1, DataType::F32, DataLayout::NHWC),
                                                          TensorInfo(TensorShape(24U, 3U, 3U), 1, DataType::F32, DataLayout::NHWC),
                                                          TensorInfo(TensorShape(24U, 4U, 3U), 1, DataType::F32, DataLayout::NHWC),
                                                        })),
                framework::dataset::make("BiasesInfo", { TensorInfo(TensorShape(2U), 1, DataType::F32, DataLayout::NHWC),
                                                         TensorInfo(TensorShape(2U), 1, DataType::F32, DataLayout::NHWC),
                                                         TensorInfo(TensorShape(2U), 1, DataType::F32, DataLayout::NHWC),
                                                         TensorInfo(TensorShape(4U), 1, DataType::F32, DataLayout::NHWC),
                                                         TensorInfo(TensorShape(2U, 2U), 1, DataType::F32, DataLayout::NHWC),
                                                         TensorInfo(TensorShape(16U), 1, DataType::F32, DataLayout::NHWC),
                                                         TensorInfo(TensorShape(16U), 1, DataType::F32, DataLayout::NHWC),
                                                         TensorInfo(TensorShape(24U), 1, DataType::S32, DataLayout::NHWC),
                                                         TensorInfo(TensorShape(24U), 1, DataType::S32, DataLayout::NHWC),
                                                         TensorInfo(TensorShape(24U), 1, DataType::S32, DataLayout::NHWC),
                                                         TensorInfo(TensorShape(24U), 1, DataType::S32, DataLayout::NHWC),
                                                         TensorInfo(TensorShape(24U), 1, DataType::S32, DataLayout::NHWC),
                                                         TensorInfo(TensorShape(24U), 1, DataType::S32, DataLayout::NHWC),
                                                         TensorInfo(TensorShape(24U), 1, DataType::S32, DataLayout::NHWC),
                                                         TensorInfo(TensorShape(24U), 1, DataType::S32, DataLayout::NHWC),
                                                         TensorInfo(TensorShape(24U), 1, DataType::S32, DataLayout::NHWC),
                                                         TensorInfo(TensorShape(24U), 1, DataType::S32, DataLayout::NHWC),
                                                         TensorInfo(TensorShape(24U), 1, DataType::S32, DataLayout::NHWC),
                                                         TensorInfo(TensorShape(24U), 1, DataType::S32, DataLayout::NHWC),
                                                         TensorInfo(TensorShape(24U), 1, DataType::S32, DataLayout::NCHW),
                                                         TensorInfo(TensorShape(24U), 1, DataType::F32, DataLayout::NHWC),
                                                         TensorInfo(TensorShape(24U), 1, DataType::F32, DataLayout::NHWC),
                                                         TensorInfo(TensorShape(24U), 1, DataType::F32, DataLayout::NHWC),
                                                         TensorInfo(TensorShape(24U), 1, DataType::F32, DataLayout::NHWC),
                                                         TensorInfo(TensorShape(24U), 1, DataType::F32, DataLayout::NHWC),
                                                       })),
                framework::dataset::make("Padding", {  Padding2D(0, 0, 0, 0),
                                                       Padding2D(0, 0, 0, 0),
                                                       Padding2D(0, 0, 0, 0),
                                                       Padding2D(0, 0, 0, 0),
                                                       Padding2D(0, 0, 0, 0),
                                                       Padding2D(0, 0, 0, 0),
                                                       Padding2D(0, 0, 0, 0),
                                                       Padding2D(1, 1, 0, 0),
                                                       Padding2D(1, 1, 0, 0),
                                                       Padding2D(1, 1, 0, 0),
                                                       Padding2D(1, 1, 0, 0),
                                                       Padding2D(1, 1, 0, 0),
                                                       Padding2D(1, 1, 0, 0),
                                                       Padding2D(1, 1, 0, 0),
                                                       Padding2D(1, 1, 0, 0),
                                                       Padding2D(1, 1, 0, 0),
                                                       Padding2D(1, 1, 0, 0),
                                                       Padding2D(1, 1, 0, 0),
                                                       Padding2D(1, 1, 0, 0),
                                                       Padding2D(1, 1, 0, 0),
                                                       Padding2D(1, 1, 0, 0),
                                                       Padding2D(1, 1, 0, 0),
                                                       Padding2D(2, 1, 2, 1),
                                                       Padding2D(2, 1, 2, 1),
                                                       Padding2D(2, 1, 2, 1),
                                                      })),
                framework::dataset::make("Stride", {   Size2D(1, 1),
                                                       Size2D(1, 1),
                                                       Size2D(1, 1),
                                                       Size2D(1, 1),
                                                       Size2D(1, 1),
                                                       Size2D(1, 1),
                                                       Size2D(1, 1),
                                                       Size2D(1, 1),
                                                       Size2D(1, 1),
                                                       Size2D(1, 1),
                                                       Size2D(1, 1),
                                                       Size2D(1, 1),
                                                       Size2D(1, 1),
                                                       Size2D(1, 1),
                                                       Size2D(1, 1),
                                                       Size2D(1, 1),
                                                       Size2D(1, 1),
                                                       Size2D(1, 1),
                                                       Size2D(1, 1),
                                                       Size2D(1, 1),
                                                       Size2D(1, 1),
                                                       Size2D(1, 1),
                                                       Size2D(1, 1),
                                                       Size2D(2, 3),
                                                       Size2D(2, 3),
                                                      })),
                framework::dataset::make("DepthMultiplier", { 1,
                                                              1,
                                                              3,
                                                              1,
                                                              1,
                                                              2,
                                                              2,
                                                              3,
                                                              3,
                                                              3,
                                                              3,
                                                              3,
                                                              3,
                                                              3,
                                                              3,
                                                              3,
                                                              3,
                                                              3,
                                                              3,
                                                              3,
                                                              3,
                                                              3,
                                                              3,
                                                              3,
                                                              3,
                                                             })),
                       framework::dataset::make("Dilation", { Size2D(1U, 1U),
                                                              Size2D(1U, 1U),
                                                              Size2D(1U, 1U),
                                                              Size2D(1U, 1U),
                                                              Size2D(1U, 1U),
                                                              Size2D(0U, 1U),
                                                              Size2D(1U, 1U),
                                                              Size2D(1U, 1U),
                                                              Size2D(1U, 1U),
                                                              Size2D(1U, 1U),
                                                              Size2D(1U, 1U),
                                                              Size2D(1U, 1U),
                                                              Size2D(1U, 1U),
                                                              Size2D(1U, 1U),
                                                              Size2D(1U, 1U),
                                                              Size2D(1U, 1U),
                                                              Size2D(1U, 1U),
                                                              Size2D(1U, 1U),
                                                              Size2D(1U, 1U),
                                                              Size2D(1U, 1U),
                                                              Size2D(1U, 1U),
                                                              Size2D(1U, 1U),
                                                              Size2D(1U, 1U),
                                                              Size2D(1U, 1U),
                                                              Size2D(2U, 3U),
                                                             })),
                framework::dataset::make("Expected", { false, false, false, false, false, false, true, false,
                                                       false, false, false, false, false, false, false, false, false, false,
                                                       false, false, true, false, true, true, true })),
                input_info, weights_info, biases_info, padding, stride, depth_multiplier, dilation, expected)
{
    CLCompileContext cl_compile_ctx = CLKernelLibrary::get().get_compile_context();
    GpuWorkloadContext context = GpuWorkloadContext{ &cl_compile_ctx };
    GpuWorkloadSketch sketch{ &context };

    const TensorInfo sketch_input_info   = context.create_tensor_info(input_info);
    const TensorInfo sketch_weights_info = context.create_tensor_info(weights_info);
    const TensorInfo sketch_biases_info  = context.create_tensor_info(biases_info);

    DepthwiseConv2dAttributes attributes {};
    attributes.pad(padding)
              .stride(stride)
              .dilation(dilation)
              .depth_multiplier(depth_multiplier);

    const Status status = GpuDepthwiseConv2d::validate_op(sketch, &sketch_input_info, &sketch_weights_info, &sketch_biases_info, attributes);
    const bool res = bool(status);
    ARM_COMPUTE_EXPECT(res == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using DynamicFusionGpuDepthwiseConv2dFixture = DynamicFusionGpuDepthwiseConv2dValidationFixture<CLTensor, CLAccessor, GpuDepthwiseConv2d, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
TEST_SUITE(W3x3)
FIXTURE_DATA_TEST_CASE(RunSmall, DynamicFusionGpuDepthwiseConv2dFixture<half>, framework::DatasetMode::ALL,
                       combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset3x3(),
                                               depth_multipliers),
                                       framework::dataset::make("DataType", DataType::F16)),
                               framework::dataset::make("DataLayout", DataLayout::NHWC)))
{
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, DynamicFusionGpuDepthwiseConv2dFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeDepthwiseConvolutionLayerDataset3x3(),
                                                                                                                        large_depth_multipliers),
                                                                                                                        framework::dataset::make("DataType", DataType::F16)),
                                                                                                                        framework::dataset::make("DataLayout", DataLayout::NHWC)))
{
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
#ifndef ACL_INTERNAL_TEST_CKW_IN_DF // Do not include this test as dilation not supported yet in DepthwiseConv2d CKW kernel
TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE(RunSmall, DynamicFusionGpuDepthwiseConv2dFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset3x3(),
                                                                                                                    depth_multipliers),
                                                                                                                    framework::dataset::make("DataType", DataType::F16)),
                                                                                                                    framework::dataset::make("DataLayout", { DataLayout::NHWC })))
{
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, DynamicFusionGpuDepthwiseConv2dFixture<half>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset3x3(),
                                               large_depth_multipliers),
                                       framework::dataset::make("DataType", DataType::F16)),
                               framework::dataset::make("DataLayout", { DataLayout::NHWC })))
{
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END() // Dilation
#endif           // ACL_INTERNAL_TEST_CKW_IN_DF
TEST_SUITE_END() // W3x3

TEST_SUITE(Generic)
FIXTURE_DATA_TEST_CASE(RunSmall, DynamicFusionGpuDepthwiseConv2dFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset(),
                                                                                                                    depth_multipliers),
                                                                                                                    framework::dataset::make("DataType", DataType::F16)),
                                                                                                                    framework::dataset::make("DataLayout", { DataLayout::NHWC })))
{
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
FIXTURE_DATA_TEST_CASE(RunLarge, DynamicFusionGpuDepthwiseConv2dFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeDepthwiseConvolutionLayerDataset(),
                                                                                                                        large_depth_multipliers),
                                                                                                                        framework::dataset::make("DataType", DataType::F16)),
                                                                                                                        framework::dataset::make("DataLayout", { DataLayout::NHWC })))
{
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
#ifndef ACL_INTERNAL_TEST_CKW_IN_DF // Do not include this test as dilation not supported yet in DepthwiseConv2d CKW kernel
TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE(RunSmall, DynamicFusionGpuDepthwiseConv2dFixture<half>, framework::DatasetMode::ALL, combine(combine(combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset(),
                                                                                                                    depth_multipliers),
                                                                                                                    framework::dataset::make("DataType", DataType::F16)),
                                                                                                                    framework::dataset::make("DataLayout", { DataLayout::NHWC })))
{
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
FIXTURE_DATA_TEST_CASE(RunLarge, DynamicFusionGpuDepthwiseConv2dFixture<half>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset(),
                                               large_depth_multipliers),
                                       framework::dataset::make("DataType", DataType::F16)),
                               framework::dataset::make("DataLayout", { DataLayout::NHWC })))
{
    validate(CLAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
TEST_SUITE_END() // Dilation
#endif           // ACL_INTERNAL_TEST_CKW_IN_DF
TEST_SUITE_END() // Generic
TEST_SUITE_END() // FP16

TEST_SUITE(FP32)
TEST_SUITE(W3x3)
FIXTURE_DATA_TEST_CASE(RunSmall, DynamicFusionGpuDepthwiseConv2dFixture<float>, framework::DatasetMode::ALL,
                       combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset3x3(),
                                               depth_multipliers),
                                       framework::dataset::make("DataType", DataType::F32)),
                               framework::dataset::make("DataLayout", DataLayout::NHWC)))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, DynamicFusionGpuDepthwiseConv2dFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeDepthwiseConvolutionLayerDataset3x3(),
                                               large_depth_multipliers),
                                       framework::dataset::make("DataType", DataType::F32)),
                               framework::dataset::make("DataLayout", DataLayout::NHWC)))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

#ifndef ACL_INTERNAL_TEST_CKW_IN_DF // Do not include this test as dilation not supported yet in DepthwiseConv2d CKW kernel
TEST_SUITE(Dilation)

FIXTURE_DATA_TEST_CASE(RunSmall, DynamicFusionGpuDepthwiseConv2dFixture<float>, framework::DatasetMode::ALL,
                       combine(combine(combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset3x3(),
                                               depth_multipliers),
                                       framework::dataset::make("DataType", DataType::F32)),
                               framework::dataset::make("DataLayout", DataLayout::NHWC)))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, DynamicFusionGpuDepthwiseConv2dFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset3x3(),
                                               large_depth_multipliers),
                                       framework::dataset::make("DataType", DataType::F32)),
                               framework::dataset::make("DataLayout", DataLayout::NHWC)))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // Dilation
#endif           // ACL_INTERNAL_TEST_CKW_IN_DF
TEST_SUITE_END() // W3x3

TEST_SUITE(Generic)
FIXTURE_DATA_TEST_CASE(RunSmall, DynamicFusionGpuDepthwiseConv2dFixture<float>, framework::DatasetMode::ALL,
                       combine(combine(combine(datasets::SmallDepthwiseConvolutionLayerDataset(),
                                               depth_multipliers),
                                       framework::dataset::make("DataType", DataType::F32)),
                               framework::dataset::make("DataLayout", { DataLayout::NHWC })))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, DynamicFusionGpuDepthwiseConv2dFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeDepthwiseConvolutionLayerDataset(),
                                               large_depth_multipliers),
                                       framework::dataset::make("DataType", DataType::F32)),
                               framework::dataset::make("DataLayout", { DataLayout::NHWC })))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLargeKernelSize, DynamicFusionGpuDepthwiseConv2dFixture<float>, framework::DatasetMode::ALL,
                       combine(combine(combine(datasets::LargeKernelSizeDepthwiseConvolutionLayerNHWCDataset(),
                                               framework::dataset::make("DepthMultiplier", { 1 })),
                                       framework::dataset::make("DataType", DataType::F32)),
                               framework::dataset::make("DataLayout", { DataLayout::NHWC })))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}

#ifndef ACL_INTERNAL_TEST_CKW_IN_DF // Do not include this test as dilation not supported yet in DepthwiseConv2d CKW kernel
TEST_SUITE(Dilation)
FIXTURE_DATA_TEST_CASE(RunSmall, DynamicFusionGpuDepthwiseConv2dFixture<float>, framework::DatasetMode::ALL, combine(combine(combine(datasets::SmallDepthwiseDilatedConvolutionLayerDataset(),
                                                                                                                     depth_multipliers),
                                                                                                                     framework::dataset::make("DataType", DataType::F32)),
                                                                                                                     framework::dataset::make("DataLayout", { DataLayout::NHWC })))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, DynamicFusionGpuDepthwiseConv2dFixture<float>, framework::DatasetMode::NIGHTLY,
                       combine(combine(combine(datasets::LargeDepthwiseDilatedConvolutionLayerDataset3x3(),
                                               large_depth_multipliers),
                                       framework::dataset::make("DataType", DataType::F32)),
                               framework::dataset::make("DataLayout", { DataLayout::NHWC })))
{
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // Dilation
#endif           // ACL_INTERNAL_TEST_CKW_IN_DF
TEST_SUITE_END() // Generic
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float
TEST_SUITE_END() // DEPTHWISE_CONV2D
TEST_SUITE_END() // DYNAMIC_FUSION
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
