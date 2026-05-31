/*
 * Copyright (c) 2021, 2024-2026 Arm Limited.
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

#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/validation/fixtures/DirectConvolution3DFixture.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
using framework::dataset::make;

namespace
{
#ifdef ARM_COMPUTE_ENABLE_FP16
const RelativeTolerance<half_float::half>
    rel_tolerance_f16(half_float::half(0.2f));               /**< Relative tolerance value for FP16 types */
const AbsoluteTolerance<float> abs_tolerance_f16(0.2f);      /**< Absolute tolerance for FP16 types */
constexpr float                tolerance_num = 0.07f;        /**< Tolerance number for the FP16 implementation */
#endif                                                       /* ARM_COMPUTE_ENABLE_FP16 */
constexpr AbsoluteTolerance<float>   tolerance_fp32(0.001f); /**< Tolerance for floating point tests */
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(1);   /**< Tolerance for quantized tests */

/** Activation function Dataset*/
const auto ActivationFunctionsDataset =
    make("ActivationInfo",
         {ActivationLayerInfo(), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 0.5f)});

const auto data_precommit = combine(zip(datasets::SmallDirectConv3DShapes(),
                                        make("StrideX", {1, 5, 8}),
                                        make("StrideY", {1, 2, 3}),
                                        make("StrideZ", {1, 2, 1}),
                                        make("PadX", {0, 1, 2}),
                                        make("PadY", {0, 2, 1}),
                                        make("PadZ", {0, 3, 5}),
                                        make("KernelWidth", {3, 5, 9}),
                                        make("KernelHeight", {2, 1, 3}),
                                        make("KernelDepth", {1, 2, 3}),
                                        make("NumKernels", {2, 3, 8})),
                                    make("HasBias", {true, false}),
                                    ActivationFunctionsDataset);
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(Convolution3D)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(
        make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U, 4U), 1U, DataType::F32, DataLayout::NDHWC), // Mismatching data type input/weights
                                                TensorInfo(TensorShape(27U, 13U, 2U, 4U), 1U, DataType::F32, DataLayout::NDHWC), // Mismatching input feature maps
                                                TensorInfo(TensorShape(27U, 13U, 2U, 4U), 1U, DataType::F32, DataLayout::NDHWC), // Invalid weights dimensions
                                                TensorInfo(TensorShape(27U, 13U, 2U, 4U), 1U, DataType::F32, DataLayout::NHWC), // Invalid data layout
                                                TensorInfo(TensorShape(27U, 13U, 2U, 4U), 1U, DataType::F32, DataLayout::NDHWC), // Invalid biases size
                                                TensorInfo(TensorShape(27U, 13U, 2U, 4U), 1U, DataType::F32, DataLayout::NDHWC), // Invalid biases dimensions
                                                TensorInfo(TensorShape(27U, 13U, 2U, 4U), 1U, DataType::F32, DataLayout::NDHWC), // Invalid output size
                                                TensorInfo(TensorShape(27U, 13U, 2U, 4U), 1U, DataType::U32, DataLayout::NDHWC), // Invalid data type
                                              }),
        make("WeightsInfo",{ TensorInfo(TensorShape(4U, 3U, 3U, 3U, 2U), 1U, DataType::F16),
                                                 TensorInfo(TensorShape(4U, 3U, 3U, 3U, 3U), 1U, DataType::F32),
                                                 TensorInfo(TensorShape(4U, 3U, 3U, 3U, 2U, 3U), 1U, DataType::F32),
                                                 TensorInfo(TensorShape(4U, 3U, 3U, 3U, 2U), 1U, DataType::F32),
                                                 TensorInfo(TensorShape(4U, 3U, 3U, 3U, 2U), 1U, DataType::F32),
                                                 TensorInfo(TensorShape(4U, 3U, 3U, 3U, 2U), 1U, DataType::F32),
                                                 TensorInfo(TensorShape(4U, 3U, 3U, 3U, 2U), 1U, DataType::F32),
                                                 TensorInfo(TensorShape(4U, 3U, 3U, 3U, 2U), 1U, DataType::U32),
                                              }),
        make("BiasesInfo",{ TensorInfo(TensorShape(4U), 1U, DataType::F32),
                                                TensorInfo(TensorShape(4U), 1U, DataType::F32),
                                                TensorInfo(TensorShape(4U), 1U, DataType::F32),
                                                TensorInfo(TensorShape(4U), 1U, DataType::F32),
                                                TensorInfo(TensorShape(3U), 1U, DataType::F32),
                                                TensorInfo(TensorShape(4U, 2U), 1U, DataType::F32),
                                                TensorInfo(TensorShape(4U), 1U, DataType::F32),
                                                TensorInfo(TensorShape(4U), 1U, DataType::F32),
                                              }),
        make("OutputInfo",{ TensorInfo(TensorShape(25U, 11U, 4U), 1U, DataType::F32),
                                                TensorInfo(TensorShape(25U, 11U, 4U), 1U, DataType::F32),
                                                TensorInfo(TensorShape(25U, 11U, 4U), 1U, DataType::F32),
                                                TensorInfo(TensorShape(25U, 11U, 4U), 1U, DataType::F32),
                                                TensorInfo(TensorShape(25U, 11U, 4U), 1U, DataType::F32),
                                                TensorInfo(TensorShape(25U, 11U, 4U), 1U, DataType::F32),
                                                TensorInfo(TensorShape(26U, 11U, 4U), 1U, DataType::F32),
                                                TensorInfo(TensorShape(25U, 11U, 4U), 1U, DataType::U32),
                                              }),
        make("Expected", { false, false, false, false, false, false, false, false})
        ),
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
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEDirectConvolution3DFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(data_precommit,
                               make("DataType", DataType::F32),
                               make("DataLayout", {DataLayout::NDHWC})))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_fp32);
}
TEST_SUITE_END() // FP32

#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEDirectConvolution3DFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(data_precommit,
                               make("DataType", DataType::F16),
                               make("DataLayout", {DataLayout::NDHWC})))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference, rel_tolerance_f16, tolerance_num, abs_tolerance_f16);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
TEST_SUITE_END() // FP16
#endif           /* ARM_COMPUTE_ENABLE_FP16 */

TEST_SUITE_END() // Float

template <typename T>
using NEDirectConvolution3DQuantizedFixture =
    DirectConvolution3DValidationQuantizedFixture<Tensor, Accessor, NEConv3D, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEDirectConvolution3DQuantizedFixture<uint8_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(zip(make("InputShape",
                                        {TensorShape(7U, 5U, 3U, 13U, 3U), TensorShape(15U, 7U, 11U, 7U),
                                         TensorShape(19U, 5U, 16U, 4U), TensorShape(13U, 5U, 17U, 2U)}),
                                   make("StrideX", {1, 3, 2, 1}),
                                   make("StrideY", {2, 1, 3, 1}),
                                   make("StrideZ", {3, 2, 1, 1}),
                                   make("PadX", {0, 2, 1, 0}),
                                   make("PadY", {1, 0, 2, 0}),
                                   make("PadZ", {2, 1, 0, 0}),
                                   make("KernelWidth", {3, 7, 5, 1}),
                                   make("KernelHeight", {5, 3, 7, 1}),
                                   make("KernelDepth", {7, 5, 3, 1}),
                                   make("NumKernels", {5, 3, 1, 11}),
                                   make("HasBias", {true, true, true, false})),
                               make("Activation", ActivationLayerInfo()),
                               make("DataType", DataType::QASYMM8),
                               make("DataLayout", DataLayout::NDHWC),
                               make("SrcQuantizationInfo", QuantizationInfo(0.1f, 10)),
                               make("WeightsQuantizationInfo", QuantizationInfo(0.3f, 20)),
                               make("DstQuantizationInfo", QuantizationInfo(0.2f, 5))))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}

TEST_SUITE_END() // QASYMM8

TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEDirectConvolution3DQuantizedFixture<int8_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(zip(make("InputShape",
                                        {TensorShape(7U, 5U, 3U, 13U, 3U), TensorShape(15U, 7U, 11U, 7U),
                                         TensorShape(19U, 5U, 16U, 4U), TensorShape(13U, 5U, 17U, 2U)}),
                                   make("StrideX", {1, 3, 2, 1}),
                                   make("StrideY", {2, 1, 3, 1}),
                                   make("StrideZ", {3, 2, 1, 1}),
                                   make("PadX", {0, 2, 1, 0}),
                                   make("PadY", {1, 0, 2, 0}),
                                   make("PadZ", {2, 1, 0, 0}),
                                   make("KernelWidth", {3, 7, 5, 1}),
                                   make("KernelHeight", {5, 3, 7, 1}),
                                   make("KernelDepth", {7, 5, 3, 1}),
                                   make("NumKernels", {5, 3, 1, 11}),
                                   make("HasBias", {true, true, true, false})),
                               make("Activation", ActivationLayerInfo()),
                               make("DataType", DataType::QASYMM8_SIGNED),
                               make("DataLayout", DataLayout::NDHWC),
                               make("SrcQuantizationInfo", QuantizationInfo(0.1f, 10)),
                               make("WeightsQuantizationInfo", QuantizationInfo(0.3f, 20)),
                               make("DstQuantizationInfo", QuantizationInfo(0.2f, 5))))
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}

TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // Convolution3D
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
