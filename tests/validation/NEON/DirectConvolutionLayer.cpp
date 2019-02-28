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
#include "arm_compute/runtime/NEON/functions/NEDirectConvolutionLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
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
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
const RelativeTolerance<half_float::half> rel_tolerance_f16(half_float::half(0.2f)); /**< Relative tolerance value for FP16 types */
const AbsoluteTolerance<float>            abs_tolerance_f16(0.2f);                   /**< Absolute tolerance for FP16 types */
constexpr float                           tolerance_num = 0.07f;                     /**< Tolerance number for the FP16 implementation */
#endif                                                                               /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
constexpr AbsoluteTolerance<float> tolerance_fp32(0.001f);                           /**< Tolerance for floating point tests */

/** Direct convolution data set.for FP32 */
const auto data_pad_f32 = concat(concat(combine(framework::dataset::make("PadX", { 0, 1 }),
                                                combine(framework::dataset::make("PadY", { 0, 1 }),
                                                        framework::dataset::make("KernelSize", 3))),
                                        combine(framework::dataset::make("PadX", { 0, 2 }),
                                                combine(framework::dataset::make("PadY", { 0, 2 }),
                                                        framework::dataset::make("KernelSize", 3)))),
                                 combine(framework::dataset::make("PadX", { 0, 3 }),
                                         combine(framework::dataset::make("PadY", { 0, 3 }),
                                                 framework::dataset::make("KernelSize", 5))));

/** Direct convolution data set.for FP16 */
const auto data_pad_f16 = concat(combine(framework::dataset::make("PadX", { 0, 1 }),
                                         combine(framework::dataset::make("PadY", { 0, 1 }),
                                                 framework::dataset::make("KernelSize", 3))),
                                 combine(framework::dataset::make("PadX", { 0 }),
                                         combine(framework::dataset::make("PadY", { 0 }),
                                                 framework::dataset::make("KernelSize", 1))));

const auto data_f32 = combine(datasets::SmallDirectConvolutionShapes(),
                              combine(framework::dataset::make("StrideX", { 1, 2, 3 }),
                                      combine(framework::dataset::make("StrideY", { 1, 2, 3 }),
                                              data_pad_f32)));

const auto data_f16 = combine(datasets::SmallDirectConvolutionShapes(),
                              combine(framework::dataset::make("StrideX", { 1, 2, 3 }),
                                      combine(framework::dataset::make("StrideY", { 1, 2, 3 }),
                                              data_pad_f16)));

const auto data = combine(datasets::SmallDirectConvolutionShapes(),
                          combine(framework::dataset::make("StrideX", { 1 }),
                                  combine(framework::dataset::make("StrideY", { 1 }),
                                          combine(framework::dataset::make("PadX", { 1 }),
                                                  combine(framework::dataset::make("PadY", { 1 }),
                                                          framework::dataset::make("KernelSize", 3))))));

const auto data_f32_nightly = combine(data_f32, framework::dataset::make("NumKernels", { 1, 4 }));
const auto data_f16_nightly = combine(data_f16, framework::dataset::make("NumKernels", { 1, 4 }));

const auto data_precommit = combine(data, framework::dataset::make("NumKernels", { 1 }));

/** Activation function Dataset*/
const auto ActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{ ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 0.5f) });
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(DirectConvolutionLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(
        framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Mismatching data type input/weights
                                                TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Mismatching input feature maps
                                                TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Unsupported kernel width
                                                TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Non-rectangular weights dimensions
                                                TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Invalid weights dimensions
                                                TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Invalid stride
                                                TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Invalid biases size
                                                TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Invalid biases dimensions
                                                TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Invalid output size
                                              }),
        framework::dataset::make("WeightsInfo",{ TensorInfo(TensorShape(3U, 3U, 2U, 4U), 1, DataType::F16),
                                                 TensorInfo(TensorShape(3U, 3U, 3U, 4U), 1, DataType::F32),
                                                 TensorInfo(TensorShape(9U, 9U, 2U, 4U), 1, DataType::F32),
                                                 TensorInfo(TensorShape(5U, 3U, 2U, 4U), 1, DataType::F32),
                                                 TensorInfo(TensorShape(3U, 3U, 2U, 4U, 3U), 1, DataType::F32),
                                                 TensorInfo(TensorShape(3U, 3U, 2U, 4U), 1, DataType::F32),
                                                 TensorInfo(TensorShape(3U, 3U, 2U, 4U), 1, DataType::F32),
                                                 TensorInfo(TensorShape(3U, 3U, 2U, 4U), 1, DataType::F32),
                                                 TensorInfo(TensorShape(3U, 3U, 2U, 4U), 1, DataType::F32),
                                              })),
        framework::dataset::make("BiasesInfo",{ TensorInfo(TensorShape(4U), 1, DataType::F32),
                                                TensorInfo(TensorShape(4U), 1, DataType::F32),
                                                TensorInfo(TensorShape(4U), 1, DataType::F32),
                                                TensorInfo(TensorShape(4U), 1, DataType::F32),
                                                TensorInfo(TensorShape(4U), 1, DataType::F32),
                                                TensorInfo(TensorShape(4U), 1, DataType::F32),
                                                TensorInfo(TensorShape(3U), 1, DataType::F32),
                                                TensorInfo(TensorShape(4U, 2U), 1, DataType::F32),
                                                TensorInfo(TensorShape(4U), 1, DataType::F32),
                                              })),
        framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(25U, 11U, 4U), 1, DataType::F32),
                                                TensorInfo(TensorShape(25U, 11U, 4U), 1, DataType::F32),
                                                TensorInfo(TensorShape(25U, 11U, 4U), 1, DataType::F32),
                                                TensorInfo(TensorShape(25U, 11U, 4U), 1, DataType::F32),
                                                TensorInfo(TensorShape(25U, 11U, 4U), 1, DataType::F32),
                                                TensorInfo(TensorShape(25U, 11U, 4U), 1, DataType::F32),
                                                TensorInfo(TensorShape(25U, 11U, 4U), 1, DataType::F32),
                                                TensorInfo(TensorShape(25U, 11U, 4U), 1, DataType::F32),
                                                TensorInfo(TensorShape(26U, 11U, 4U), 1, DataType::F32),
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
                                               })),
                                                       framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)
})),
        framework::dataset::make("Expected", { false, false, false, false, false, false, false, false, false })),
        input_info, weights_info, biases_info, output_info, conv_info, act_info, expected)
{
        bool is_valid = bool(NEDirectConvolutionLayer::validate(&input_info.clone()->set_is_resizable(false), &weights_info.clone()->set_is_resizable(false), &biases_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), conv_info, act_info));
        ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

//TODO(COMPMID-415): Configuration tests?

template <typename T>
using NEDirectConvolutionLayerFixture = DirectConvolutionValidationFixture<Tensor, Accessor, NEDirectConvolutionLayer, T>;

TEST_SUITE(Float)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDirectConvolutionLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(data_precommit, framework::dataset::make("DataType",
                                                                                                                   DataType::F16)),
                                                                                                                   ActivationFunctionsDataset),
                                                                                                                   framework::dataset::make("DataLayout", DataLayout::NCHW)))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDirectConvolutionLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(data_f16_nightly, framework::dataset::make("DataType", DataType::F16)),
                                                                                                                 ActivationFunctionsDataset),
                                                                                                                 framework::dataset::make("DataLayout", DataLayout::NCHW)))
{
    // Validate output
    validate(Accessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_f16);
}
TEST_SUITE_END() // FP16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDirectConvolutionLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(data_precommit, framework::dataset::make("DataType",
                                                                                                                    DataType::F32)),
                                                                                                                    ActivationFunctionsDataset),
                                                                                                                    framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDirectConvolutionLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(data_f32_nightly, framework::dataset::make("DataType",
                                                                                                                  DataType::F32)),
                                                                                                                  ActivationFunctionsDataset),
                                                                                                                  framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

const auto QuantizedActivationFunctionsDataset = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f)
});

TEST_SUITE_END() // DirectConvolutionLayer
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
