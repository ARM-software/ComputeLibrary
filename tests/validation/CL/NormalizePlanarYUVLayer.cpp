/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLNormalizePlanarYUVLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/RandomNormalizePlanarYUVLayerDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/NormalizePlanarYUVLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr RelativeTolerance<float> tolerance_f16(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F16 */
constexpr RelativeTolerance<float> tolerance_f32(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
constexpr AbsoluteTolerance<float> tolerance_qasymm8(1);  /**< Tolerance value for comparing reference's output against implementation's output for quantized data types */
} // namespace

TEST_SUITE(CL)
TEST_SUITE(NormalizePlanarYUVLayer)

template <typename T>
using CLNormalizePlanarYUVLayerFixture = NormalizePlanarYUVLayerValidationFixture<CLTensor, CLAccessor, CLNormalizePlanarYUVLayer, T>;

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
                    framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Mismatching data types
                        TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F16),     // Window shrink
                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),      // Unsupported data type
                        TensorInfo(TensorShape(32U, 16U, 8U), 1, DataType::F16),
                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F16),     // Mismatching mean and sd shapes
                        TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Mismatching shapes
                        }),
                    framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F16),
                        TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F16),
                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                        TensorInfo(TensorShape(32U, 16U, 8U), 1, DataType::F16),
                        TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F16),
                        TensorInfo(TensorShape(30U, 11U, 2U), 1, DataType::F32),
                        })),
                framework::dataset::make("MSTDInfo",{ TensorInfo(TensorShape(2U), 1, DataType::F16),
                    TensorInfo(TensorShape(2U), 1, DataType::F16),
                    TensorInfo(TensorShape(2U), 1, DataType::U8),
                    TensorInfo(TensorShape(8U), 1, DataType::F16),
                    TensorInfo(TensorShape(6U), 1, DataType::F16),
                    TensorInfo(TensorShape(2U), 1, DataType::F32),
                    })),
                    framework::dataset::make("Expected", { false, false, false, true, false, false })),
                    input_info, output_info, msd_info, expected)
{
    const auto &mean_info = msd_info;
    const auto &sd_info   = msd_info;
    bool has_error = bool(CLNormalizePlanarYUVLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), &mean_info.clone()->set_is_resizable(false), &sd_info.clone()->set_is_resizable(false)));
    ARM_COMPUTE_EXPECT(has_error == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(Random, CLNormalizePlanarYUVLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::RandomNormalizePlanarYUVLayerDataset(),
                                                                                                                  framework::dataset::make("DataType", DataType::F16)),
                                                                                                                  framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16, 0);
}
TEST_SUITE_END() // FP16

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(Random, CLNormalizePlanarYUVLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::RandomNormalizePlanarYUVLayerDataset(),
                                                                                                                   framework::dataset::make("DataType", DataType::F32)),
                                                                                                                   framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

template <typename T>
using CLNormalizePlanarYUVLayerQuantizedFixture = NormalizePlanarYUVLayerValidationQuantizedFixture<CLTensor, CLAccessor, CLNormalizePlanarYUVLayer, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(Random, CLNormalizePlanarYUVLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::RandomNormalizePlanarYUVLayerDataset(),
                       framework::dataset::make("DataType", DataType::QASYMM8)),
                       framework::dataset::make("DataLayout", { DataLayout::NHWC })),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.1f, 128.0f) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8, 0);
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(Random, CLNormalizePlanarYUVLayerQuantizedFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::RandomNormalizePlanarYUVLayerDataset(),
                       framework::dataset::make("DataType", DataType::QASYMM8_SIGNED)),
                       framework::dataset::make("DataLayout", { DataLayout::NCHW })),
                       framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.1f, 128.0f) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8, 0);
}
TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // NormalizePlanarYUVLayer
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
