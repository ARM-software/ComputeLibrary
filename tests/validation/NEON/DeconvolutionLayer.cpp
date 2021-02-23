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
#include "arm_compute/runtime/NEON/functions/NEDeconvolutionLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/DeconvolutionLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr AbsoluteTolerance<float> tolerance_fp32(0.001f);    /**< Tolerance for floating point tests */
constexpr AbsoluteTolerance<float> tolerance_quantized(1.0f); /**< Tolerance value for comparing reference's output against implementation's output for quantized data types */
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
const RelativeTolerance<half_float::half> tolerance_fp16(half_float::half(0.2f)); /**< Relative tolerance value for comparing reference's output against implementation's output for DataType::F16 */
#endif                                                                            /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC*/
constexpr float tolerance_num = 0.07f;                                            /**< Tolerance number */

const auto data4x4 = datasets::SmallDeconvolutionShapes() * framework::dataset::make("StrideX", 1, 4) * framework::dataset::make("StrideY", 1, 4) * framework::dataset::make("PadX", 0, 3)
                     * framework::dataset::make("PadY", 0, 3) * framework::dataset::make("NumKernels", { 3 });

const auto data3x3 = datasets::SmallDeconvolutionShapes() * framework::dataset::make("StrideX", 1, 4) * framework::dataset::make("StrideY", 1, 4) * framework::dataset::make("PadX", 0, 2)
                     * framework::dataset::make("PadY", 0, 2) * framework::dataset::make("NumKernels", { 3 });

const auto data3x3_asymm = datasets::SmallDeconvolutionShapes() * framework::dataset::make("StrideX", 1, 2) * framework::dataset::make("StrideY", 1, 2) * framework::dataset::make("PadLeft", 0, 1)
                           * framework::dataset::make("PadRight", 0, 1) * framework::dataset::make("PadTop", 0, 1) * framework::dataset::make("PadBottom", 0, 1) * framework::dataset::make("NumKernels", { 3 });

const auto data9x9_small_asymm = framework::dataset::make("InputShape", TensorShape{ 10U, 10U, 1U, 1U }) *framework::dataset::make("StrideX", 2) *framework::dataset::make("StrideY",
                                 2)
                                 *framework::dataset::make("PadLeft", 3)
                                 *framework::dataset::make("PadRight", 4) *framework::dataset::make("PadTop", 3) *framework::dataset::make("PadBottom", 4) *framework::dataset::make("NumKernels", { 1 });

const auto data9x9_large_asymm = framework::dataset::make("InputShape", TensorShape{ 640U, 360U, 56U, 1U }) *framework::dataset::make("StrideX", 2) *framework::dataset::make("StrideY",
                                 2)
                                 *framework::dataset::make("PadLeft", 3)
                                 *framework::dataset::make("PadRight", 4) *framework::dataset::make("PadTop", 3) *framework::dataset::make("PadBottom", 4) *framework::dataset::make("NumKernels", { 1 });

const auto data3x3_precommit = datasets::SmallDeconvolutionShapes() * framework::dataset::make("StrideX", 1, 2) * framework::dataset::make("StrideY", 1, 2) * framework::dataset::make("PadX", 0, 2)
                               * framework::dataset::make("PadY", 0, 2) * framework::dataset::make("NumKernels", { 3 });

const auto data1x1 = datasets::SmallDeconvolutionShapes() * framework::dataset::make("StrideX", 1, 4) * framework::dataset::make("StrideY", 1, 4) * framework::dataset::make("PadX", 0, 1)
                     * framework::dataset::make("PadY", 0, 1) * framework::dataset::make("NumKernels", { 3 });

const auto data_layouts_dataset = framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC });

const auto add_bias_dataset = framework::dataset::make("AddBias", { true, false });

const auto input_qinfo_dataset = framework::dataset::make("InputQInfo",
{
    QuantizationInfo(1.f / 255.f, 0),
    QuantizationInfo(2.f, 0),
});

const auto output_qinfo_dataset = framework::dataset::make("OutputQInfo",
{
    QuantizationInfo(3.f / 255.f, 0),
    QuantizationInfo(4.f, 0),
});
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(DeconvolutionLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(
    framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),   // Mismatching data type
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),   // Invalid weights shape
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F16),   // Non supported data type
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),  // Invalid bias shape
                                            TensorInfo(TensorShape(13U, 11U, 4U, 3U), 1, DataType::F32), // Window shrink
                                            TensorInfo(TensorShape(32U, 16U, 2U), 1, DataType::F32),
                                          }),
    framework::dataset::make("WeightsInfo", { TensorInfo(TensorShape(3U, 3U, 2U, 2U), 1, DataType::F16),
                                            TensorInfo(TensorShape(3U, 3U, 2U, 4U), 1, DataType::F32),
                                            TensorInfo(TensorShape(3U, 3U, 2U, 2U), 1, DataType::F16),
                                            TensorInfo(TensorShape(3U, 2U, 2U, 2U), 1, DataType::F32),
                                            TensorInfo(TensorShape(3U, 3U, 4U), 1, DataType::F32),
                                              TensorInfo(TensorShape(1U, 1U, 2U, 4U), 1, DataType::F32),
                                          })),
    framework::dataset::make("BiasInfo",  { TensorInfo(TensorShape(1U), 1, DataType::F16),
                                            TensorInfo(TensorShape(1U), 1, DataType::F32),
                                            TensorInfo(TensorShape(1U), 1, DataType::F32),
                                            TensorInfo(TensorShape(25U, 11U), 1, DataType::F32),
                                            TensorInfo(TensorShape(1U), 1, DataType::F32),
                                            TensorInfo(TensorShape(4U), 1, DataType::F32),
                                          })),
    framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F16),
                                            TensorInfo(TensorShape(25U, 10U, 2U), 1, DataType::F32),
                                            TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32),
                                            TensorInfo(TensorShape(13U, 13U, 2U), 1, DataType::F32),
                                            TensorInfo(TensorShape(11U, 9U, 1U, 3U), 1, DataType::F32),
                                            TensorInfo(TensorShape(32U, 16U, 4U), 1, DataType::F32),
                                          })),
    framework::dataset::make("PadStrideInfo", { PadStrideInfo(1, 1, 0, 0),
                                                PadStrideInfo(1, 1, 0, 0),
                                                PadStrideInfo(1, 1, 0, 0),
                                                PadStrideInfo(1, 1, 0, 0),
                                                PadStrideInfo(1, 1, 1, 1),
                                                PadStrideInfo(1, 1, 0, 0),
                                           })),
    framework::dataset::make("Expected", { false, false, false, false, false, true })),
    input_info, weights_info, bias_info, output_info, pad_info, expected)
{
    bool is_valid = bool(NEDeconvolutionLayer::validate(&input_info.clone()->set_is_resizable(false), &weights_info.clone()->set_is_resizable(false), &bias_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), pad_info));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEDeconvolutionLayerFixture4x4 = DeconvolutionValidationFixture<Tensor, Accessor, NEDeconvolutionLayer, T, 4, 4>;

template <typename T>
using NEDeconvolutionLayerFixture3x3 = DeconvolutionValidationFixture<Tensor, Accessor, NEDeconvolutionLayer, T, 3, 3>;

template <typename T>
using NEDeconvolutionLayerAsymmFixture3x3 = DeconvolutionValidationAsymmFixture<Tensor, Accessor, NEDeconvolutionLayer, T, 3, 3>;

template <typename T>
using NEDeconvolutionLayerAsymmFixture9x9 = DeconvolutionValidationAsymmFixture<Tensor, Accessor, NEDeconvolutionLayer, T, 9, 9>;

template <typename T>
using NEDeconvolutionLayerFixture1x1 = DeconvolutionValidationFixture<Tensor, Accessor, NEDeconvolutionLayer, T, 1, 1>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
TEST_SUITE(W4x4)
FIXTURE_DATA_TEST_CASE(Run, NEDeconvolutionLayerFixture4x4<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(data4x4, framework::dataset::make("DataType", DataType::F32)),
                                                                                                                    data_layouts_dataset),
                                                                                                            add_bias_dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}
TEST_SUITE_END() // W4x4
TEST_SUITE(W3x3)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDeconvolutionLayerFixture3x3<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(data3x3_precommit, framework::dataset::make("DataType",
                                                                                                                   DataType::F32)),
                                                                                                                   data_layouts_dataset),
                                                                                                                   add_bias_dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}
FIXTURE_DATA_TEST_CASE(RunAsymm, NEDeconvolutionLayerAsymmFixture3x3<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(data3x3_asymm, framework::dataset::make("DataType",
                                                                                                                      DataType::F32)),
                                                                                                                      data_layouts_dataset),
                                                                                                                      add_bias_dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDeconvolutionLayerFixture3x3<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(data3x3, framework::dataset::make("DataType", DataType::F32)),
                                                                                                                 data_layouts_dataset),
                                                                                                                 add_bias_dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}
TEST_SUITE_END() // W3x3
TEST_SUITE(W1x1)
FIXTURE_DATA_TEST_CASE(Run, NEDeconvolutionLayerFixture1x1<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(data1x1, framework::dataset::make("DataType", DataType::F32)),
                                                                                                                    data_layouts_dataset),
                                                                                                            add_bias_dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}
TEST_SUITE_END() // W1x1
TEST_SUITE(W9x9)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDeconvolutionLayerAsymmFixture9x9<float>, framework::DatasetMode::ALL, combine(combine(combine(data9x9_small_asymm, framework::dataset::make("DataType",
                                                                                                                  DataType::F32)),
                                                                                                                  framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                                                                                                  framework::dataset::make("AddBias", { false })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDeconvolutionLayerAsymmFixture9x9<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(data9x9_large_asymm, framework::dataset::make("DataType",
                                                                                                                      DataType::F32)),
                                                                                                                      framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                                                                                                                      framework::dataset::make("AddBias", { false })))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}
TEST_SUITE_END() // W9x9
TEST_SUITE_END() // FP32

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
TEST_SUITE(W4x4)
FIXTURE_DATA_TEST_CASE(Run, NEDeconvolutionLayerFixture4x4<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(data4x4, framework::dataset::make("DataType", DataType::F16)),
                                                                                                                   data_layouts_dataset),
                                                                                                           add_bias_dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp16);
}
TEST_SUITE_END() // W4x4
TEST_SUITE(W3x3)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDeconvolutionLayerFixture3x3<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(data3x3_precommit, framework::dataset::make("DataType",
                                                                                                                  DataType::F16)),
                                                                                                                  data_layouts_dataset),
                                                                                                                  add_bias_dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDeconvolutionLayerFixture3x3<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(data3x3, framework::dataset::make("DataType", DataType::F16)),
                                                                                                                        data_layouts_dataset),
                                                                                                                add_bias_dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp16);
}
TEST_SUITE_END() // W3x3
TEST_SUITE(W1x1)
FIXTURE_DATA_TEST_CASE(Run, NEDeconvolutionLayerFixture1x1<half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(data1x1, framework::dataset::make("DataType", DataType::F16)),
                                                                                                                   data_layouts_dataset),
                                                                                                           add_bias_dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp16);
}
TEST_SUITE_END() // W1x1
TEST_SUITE_END() // FP16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE_END() // Float

template <typename T>
using NEDeconvolutionLayerQuantizedFixture4x4 = DeconvolutionValidationQuantizedFixture<Tensor, Accessor, NEDeconvolutionLayer, T, 4, 4>;

template <typename T>
using NEDeconvolutionLayerQuantizedFixture3x3 = DeconvolutionValidationQuantizedFixture<Tensor, Accessor, NEDeconvolutionLayer, T, 3, 3>;

template <typename T>
using NEDeconvolutionLayerQuantizedFixture1x1 = DeconvolutionValidationQuantizedFixture<Tensor, Accessor, NEDeconvolutionLayer, T, 1, 1>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)

TEST_SUITE(W4x4)
FIXTURE_DATA_TEST_CASE(Run, NEDeconvolutionLayerQuantizedFixture4x4<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(data4x4, framework::dataset::make("DataType",
                                                                                                                       DataType::QASYMM8)),
                                                                                                                       data_layouts_dataset),
                                                                                                                       input_qinfo_dataset),
                                                                                                                       output_qinfo_dataset),
                                                                                                                       add_bias_dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_quantized, tolerance_num);
}
TEST_SUITE_END() // W4x4

TEST_SUITE(W3x3)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDeconvolutionLayerQuantizedFixture3x3<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(combine(data3x3_precommit,
                       framework::dataset::make("DataType",
                                                DataType::QASYMM8)),
                       data_layouts_dataset),
                       input_qinfo_dataset),
                       output_qinfo_dataset),
                       add_bias_dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_quantized, tolerance_num);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDeconvolutionLayerQuantizedFixture3x3<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(data3x3,
                       framework::dataset::make("DataType",
                                                DataType::QASYMM8)),
                       data_layouts_dataset),
                       input_qinfo_dataset),
                       output_qinfo_dataset),
                       add_bias_dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_quantized, tolerance_num);
}
TEST_SUITE_END() // W3x3

TEST_SUITE(W1x1)
FIXTURE_DATA_TEST_CASE(Run, NEDeconvolutionLayerQuantizedFixture1x1<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(data1x1, framework::dataset::make("DataType",
                                                                                                                       DataType::QASYMM8)),
                                                                                                                       data_layouts_dataset),
                                                                                                                       input_qinfo_dataset),
                                                                                                                       output_qinfo_dataset),
                                                                                                                       add_bias_dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_quantized, tolerance_num);
}
TEST_SUITE_END() // W1x1

TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)

TEST_SUITE(W4x4)
FIXTURE_DATA_TEST_CASE(Run, NEDeconvolutionLayerQuantizedFixture4x4<int8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(data4x4, framework::dataset::make("DataType",
                                                                                                                      DataType::QASYMM8_SIGNED)),
                                                                                                                      data_layouts_dataset),
                                                                                                                      input_qinfo_dataset),
                                                                                                                      output_qinfo_dataset),
                                                                                                                      add_bias_dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_quantized, tolerance_num);
}
TEST_SUITE_END() // W4x4

TEST_SUITE(W3x3)
FIXTURE_DATA_TEST_CASE(RunSmall, NEDeconvolutionLayerQuantizedFixture3x3<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(combine(data3x3_precommit,
                       framework::dataset::make("DataType",
                                                DataType::QASYMM8_SIGNED)),
                       data_layouts_dataset),
                       input_qinfo_dataset),
                       output_qinfo_dataset),
                       add_bias_dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_quantized, tolerance_num);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEDeconvolutionLayerQuantizedFixture3x3<int8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(data3x3,
                       framework::dataset::make("DataType",
                                                DataType::QASYMM8_SIGNED)),
                       data_layouts_dataset),
                       input_qinfo_dataset),
                       output_qinfo_dataset),
                       add_bias_dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_quantized, tolerance_num);
}
TEST_SUITE_END() // W3x3

TEST_SUITE(W1x1)
FIXTURE_DATA_TEST_CASE(Run, NEDeconvolutionLayerQuantizedFixture1x1<int8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(combine(data1x1, framework::dataset::make("DataType",
                                                                                                                      DataType::QASYMM8_SIGNED)),
                                                                                                                      data_layouts_dataset),
                                                                                                                      input_qinfo_dataset),
                                                                                                                      output_qinfo_dataset),
                                                                                                                      add_bias_dataset))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_quantized, tolerance_num);
}
TEST_SUITE_END() // W1x1

TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // DeconvolutionLayer
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
