/*
 * Copyright (c) 2021 Arm Limited.
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
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEConv3D.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/DirectConvolution3DFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
const RelativeTolerance<half_float::half> rel_tolerance_f16(half_float::half(0.2f)); /**< Relative tolerance value for FP16 types */
const AbsoluteTolerance<float>            abs_tolerance_f16(0.2f);                   /**< Absolute tolerance for FP16 types */
constexpr float                           tolerance_num = 0.07f;                     /**< Tolerance number for the FP16 implementation */
#endif                                                                               /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
constexpr AbsoluteTolerance<float> tolerance_fp32(0.001f);                           /**< Tolerance for floating point tests */

/* The following tests are from real use-case that made DirectConvolution
 * overflows in terms of its tensor indexing. This test case is using
 * a separate tolerance due to the following reason.
 * - It has shown that it requires generally larger absolute tolerance
 *   for large numbers or larger relative tolerance for small numbers.
 * - With the first reason, since it is mainly testing index overflow,
 *   a value with a margin is used to avoid uninteded test failures
 *   during nightly.
 */
constexpr AbsoluteTolerance<float> usecase_tolerance_fp32(0.05f);

/** Activation function Dataset*/
const auto ActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 0.5f)
});

const auto data_precommit = combine(combine(zip(zip(zip(zip(zip(zip(zip(zip(zip(zip(
                                                                                    datasets::SmallDirectConv3DShapes(),
                                                                                    framework::dataset::make("StrideX", { 1, 5, 8 })),
                                                                                framework::dataset::make("StrideY", { 1, 2, 3 })),
                                                                            framework::dataset::make("StrideZ", { 1, 2, 1 })),
                                                                        framework::dataset::make("PadX", { 0, 1, 2 })),
                                                                    framework::dataset::make("PadY", { 0, 2, 1 })),
                                                                framework::dataset::make("PadZ", { 0, 3, 5 })),
                                                            framework::dataset::make("KernelWidth", { 3, 5, 9 })),
                                                        framework::dataset::make("KernelHeight", { 2, 1, 3 })),
                                                    framework::dataset::make("KernelDepth", { 1, 2, 3 })),
                                                framework::dataset::make("NumKernels", { 2, 3, 8 })),
                                            framework::dataset::make("HasBias", { true, false })),
                                    ActivationFunctionsDataset);
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(Convolution3D)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
        framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U, 4U), 1U, DataType::F32, DataLayout::NDHWC), // Mismatching data type input/weights
                                                TensorInfo(TensorShape(27U, 13U, 2U, 4U), 1U, DataType::F32, DataLayout::NDHWC), // Mismatching input feature maps
                                                TensorInfo(TensorShape(27U, 13U, 2U, 4U), 1U, DataType::F32, DataLayout::NDHWC), // Invalid weights dimensions
                                                TensorInfo(TensorShape(27U, 13U, 2U, 4U), 1U, DataType::F32, DataLayout::NHWC), // Invalid data layout
                                                TensorInfo(TensorShape(27U, 13U, 2U, 4U), 1U, DataType::F32, DataLayout::NDHWC), // Invalid biases size
                                                TensorInfo(TensorShape(27U, 13U, 2U, 4U), 1U, DataType::F32, DataLayout::NDHWC), // Invalid biases dimensions
                                                TensorInfo(TensorShape(27U, 13U, 2U, 4U), 1U, DataType::F32, DataLayout::NDHWC), // Invalid output size
                                              }),
        framework::dataset::make("WeightsInfo",{ TensorInfo(TensorShape(4U, 3U, 3U, 3U, 2U), 1U, DataType::F16),
                                                 TensorInfo(TensorShape(4U, 3U, 3U, 3U, 3U), 1U, DataType::F32),
                                                 TensorInfo(TensorShape(4U, 3U, 3U, 3U, 2U, 3U), 1U, DataType::F32),
                                                 TensorInfo(TensorShape(4U, 3U, 3U, 3U, 2U), 1U, DataType::F32),
                                                 TensorInfo(TensorShape(4U, 3U, 3U, 3U, 2U), 1U, DataType::F32),
                                                 TensorInfo(TensorShape(4U, 3U, 3U, 3U, 2U), 1U, DataType::F32),
                                                 TensorInfo(TensorShape(4U, 3U, 3U, 3U, 2U), 1U, DataType::F32),
                                              })),
        framework::dataset::make("BiasesInfo",{ TensorInfo(TensorShape(4U), 1U, DataType::F32),
                                                TensorInfo(TensorShape(4U), 1U, DataType::F32),
                                                TensorInfo(TensorShape(4U), 1U, DataType::F32),
                                                TensorInfo(TensorShape(4U), 1U, DataType::F32),
                                                TensorInfo(TensorShape(3U), 1U, DataType::F32),
                                                TensorInfo(TensorShape(4U, 2U), 1U, DataType::F32),
                                                TensorInfo(TensorShape(4U), 1U, DataType::F32),
                                              })),
        framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(25U, 11U, 4U), 1U, DataType::F32),
                                                TensorInfo(TensorShape(25U, 11U, 4U), 1U, DataType::F32),
                                                TensorInfo(TensorShape(25U, 11U, 4U), 1U, DataType::F32),
                                                TensorInfo(TensorShape(25U, 11U, 4U), 1U, DataType::F32),
                                                TensorInfo(TensorShape(25U, 11U, 4U), 1U, DataType::F32),
                                                TensorInfo(TensorShape(25U, 11U, 4U), 1U, DataType::F32),
                                                TensorInfo(TensorShape(26U, 11U, 4U), 1U, DataType::F32),
                                              })),
        framework::dataset::make("Expected", { false, false, false, false, false, false, false })),
        input_info, weights_info, biases_info, output_info, expected)
{
        const Conv3dInfo  conv3d_info(Size3D(1, 1, 1), Padding3D(0, 0, 0), ActivationLayerInfo(), Size3D(1U, 1U, 1U), DimensionRoundingType::FLOOR, false);
        bool is_valid = bool(NEConv3D::validate(&input_info.clone()->set_is_resizable(false), &weights_info.clone()->set_is_resizable(false), &biases_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), conv3d_info));
        ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEDirectConvolution3DFixture = DirectConvolution3DValidationFixture<Tensor, Accessor, NEConv3D, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDirectConvolution3DFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(data_precommit,
                                                                                                                 framework::dataset::make("DataType", DataType::F32)),
                                                                                                                 framework::dataset::make("DataLayout", { DataLayout::NDHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}
TEST_SUITE_END() // FP32

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDirectConvolution3DFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(data_precommit,
                                                                                                                        framework::dataset::make("DataType", DataType::F16)),
                                                                                                                framework::dataset::make("DataLayout", { DataLayout::NDHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_f16);
}
TEST_SUITE_END() // FP16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE_END() // Float
TEST_SUITE_END() // Convolution3D
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
