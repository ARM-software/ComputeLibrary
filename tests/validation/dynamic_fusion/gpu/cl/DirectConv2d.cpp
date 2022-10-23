/*
 * Copyright (c) 2022 Arm Limited.
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

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/dynamic_fusion/runtime/gpu/cl/ClWorkloadRuntime.h"
#include "arm_compute/dynamic_fusion/sketch/OperatorAttributes.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadSketch.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuConv2d.h"

#include "tests/AssetsLibrary.h"
#include "tests/CL/CLAccessor.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/reference/ConvolutionLayer.h"

#include "tests/datasets/SmallConvolutionLayerDataset.h"
#include "tests/validation/fixtures/dynamic_fusion/gpu/cl/DirectConv2dFixture.h"

#ifdef ARM_COMPUTE_ASSERTS_ENABLED
#include "tests/SimpleTensorPrinter.h"
#endif /* ARM_COMPUTE_ASSERTS_ENABLED */
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"
namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(DYNAMIC_FUSION)
TEST_SUITE(GPU_CONV2D)

RelativeTolerance<float>            tolerance_f32(0.01f);                 /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
RelativeTolerance<half_float::half> tolerance_f16(half_float::half(0.1)); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F16 */
constexpr float                     tolerance_num = 0.02f;                /**< Tolerance number */

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

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
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
#endif           //  __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

TEST_SUITE_END() // GPU_CONV2D
TEST_SUITE_END() // DYNAMIC_FUSION
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
