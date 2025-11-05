/*
 * Copyright (c) 2017-2021, 2023-2025 Arm Limited.
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

#include "arm_compute/runtime/experimental/operators/CpuPool2d.h"

#include "arm_compute/core/Types.h" // required for PoolingLayerInfo
#include "arm_compute/runtime/Tensor.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "tests/datasets/PoolingLayerDataset.h"
#include "tests/datasets/PoolingTypesDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/validation/fixtures/CpuPool2dFixture.h"
#include "tests/validation/Validation.h"
/*
 * Tests for arm_compute::experimental::op::CpuPool2d which is a shallow wrapper for
 * arm_compute::cpu::CpuPool2d. Any future testing to the functionalities of cpu::CpuPool2d
 * will be tested in tests/NEON/PoolingLayer.cpp given that op::CpuPool2d remain a
 * shallow wrapper.
*/
namespace arm_compute
{
namespace test
{
namespace validation
{
using framework::dataset::make;

const auto pool_data_layout_dataset = make("DataLayout", {DataLayout::NCHW, DataLayout::NHWC});

const auto SmokePoolingDatasetFP32 =
    combine(datasets::SmallNoneUnitShapes(),
            datasets::PoolingTypes(),
            make("PoolingSize", {Size2D(2, 2), Size2D(3, 3), Size2D(7, 7)}),
            make("PadStride", {PadStrideInfo(1, 1, 0, 0), PadStrideInfo(2, 2, 0, 0), PadStrideInfo(1, 2, 1, 1)}),
            make("ExcludePadding", {true, false}),
            make("DataType", DataType::F32),
            pool_data_layout_dataset);

const auto SmokePoolingDatasetQASYMM8 = combine(datasets::SmallNoneUnitShapes(),
                                                make("PoolingType", {PoolingType::MAX, PoolingType::AVG}),
                                                make("PoolingSize", {Size2D(2, 2), Size2D(3, 3)}),
                                                make("PadStride", {PadStrideInfo(1, 1, 0, 0)}),
                                                make("ExcludePadding", {false}),
                                                make("DataType", DataType::QASYMM8),
                                                pool_data_layout_dataset,
                                                make("InputQuantInfo", {QuantizationInfo(0.2f, 10)}),
                                                make("OutputQuantInfo", {QuantizationInfo(0.2f, 10)}));

/** Tolerance for float operations */
constexpr AbsoluteTolerance<float>   tolerance_f32(0.000001f);
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(
    1); /**< Tolerance value for comparing reference's output against implementation's output for unsigned 8-bit asymmetric type */

TEST_SUITE(NEON)
TEST_SUITE(OPERATORS)
TEST_SUITE(CpuPool2d)

// clang-format off
 DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(
    make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Mismatching data type
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Window shrink
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Invalid pad/size combination
                                            TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Invalid pad/size combination
                                            TensorInfo(TensorShape(15U, 13U, 5U), 1, DataType::F32),     // Non-rectangular Global Pooling
                                            TensorInfo(TensorShape(13U, 13U, 5U), 1, DataType::F32),     // Invalid output Global Pooling
                                            TensorInfo(TensorShape(13U, 13U, 5U), 1, DataType::QASYMM8), // Invalid exclude_padding = false with quantized type, no actual padding and NHWC
                                            TensorInfo(TensorShape(13U, 13U, 5U), 1, DataType::F32),
                                            TensorInfo(TensorShape(1U, 16U, 1U),  1, DataType::F32),
                                            TensorInfo(TensorShape(112, 112, 64,1), 1, DataType::F32, DataLayout::NHWC), // Mismatching number of channels
                                            TensorInfo(TensorShape(112, 112, 64,1), 1, DataType::F32, DataLayout::NHWC), // Mismatching width
                                         }),
    make("OutputInfo",{ TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F16),
                                            TensorInfo(TensorShape(25U, 10U, 2U), 1, DataType::F32),
                                            TensorInfo(TensorShape(30U, 11U, 2U), 1, DataType::F32),
                                            TensorInfo(TensorShape(25U, 16U, 2U), 1, DataType::F32),
                                            TensorInfo(TensorShape(1U, 1U, 5U), 1, DataType::F32),
                                            TensorInfo(TensorShape(2U, 2U, 5U), 1, DataType::F32),
                                            TensorInfo(TensorShape(12U, 12U, 5U), 1, DataType::QASYMM8),
                                            TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32),
                                            TensorInfo(TensorShape(1U, 15U, 1U), 1, DataType::F32),
                                            TensorInfo(TensorShape(56, 56, 64,1), 1, DataType::F32, DataLayout::NHWC),
                                            TensorInfo(TensorShape(56, 51, 64,1), 1, DataType::F32, DataLayout::NHWC),
                                           }),
    make("PoolInfo",  { PoolingLayerInfo(PoolingType::AVG, 3, DataLayout::NCHW, PadStrideInfo(1, 1, 0, 0)),
                                            PoolingLayerInfo(PoolingType::AVG, 3, DataLayout::NCHW, PadStrideInfo(1, 1, 0, 0)),
                                            PoolingLayerInfo(PoolingType::AVG, 2, DataLayout::NCHW, PadStrideInfo(1, 1, 2, 0)),
                                            PoolingLayerInfo(PoolingType::AVG, 2, DataLayout::NCHW, PadStrideInfo(1, 1, 0, 2)),
                                            PoolingLayerInfo(PoolingType::AVG, DataLayout::NCHW),
                                            PoolingLayerInfo(PoolingType::MAX, DataLayout::NCHW),
                                            PoolingLayerInfo(PoolingType::AVG, 2, DataLayout::NHWC, PadStrideInfo(), false),
                                            PoolingLayerInfo(PoolingType::AVG, DataLayout::NCHW),
                                            PoolingLayerInfo(PoolingType::MAX, 2, DataLayout::NHWC, PadStrideInfo(1, 1, 0, 0), false),
                                            PoolingLayerInfo(PoolingType::MAX,3,DataLayout::NHWC,PadStrideInfo(2,2,1,1)),
                                            PoolingLayerInfo(PoolingType::MAX,3,DataLayout::NHWC,PadStrideInfo(2,2,1,1)),
                                           }),
    make("Expected", { false, false, false, false, true, false, true, false, false, false, false})),
    input_info, output_info, pool_info, expected)
{
    bool is_valid = bool(arm_compute::experimental::op::CpuPool2d::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), pool_info));
    ARM_COMPUTE_EXPECT(is_valid == expected, framework::LogLevel::ERRORS);
}
// clang-format on

template <typename T>
using CpuPool2dQuantizedFixture =
    CpuPool2dValidationQuantizedFixture<Tensor, Accessor, arm_compute::experimental::op::CpuPool2d, T>;

template <typename T>
using CpuPool2dFP32Fixture = CpuPool2dValidationFixture<Tensor, Accessor, arm_compute::experimental::op::CpuPool2d, T>;

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(SmokeFP32,
                       CpuPool2dFP32Fixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       SmokePoolingDatasetFP32)
{
    validate(Accessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // FP32

TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(SmokeQASYMM8,
                       CpuPool2dQuantizedFixture<uint8_t>,
                       framework::DatasetMode::PRECOMMIT,
                       SmokePoolingDatasetQASYMM8)
{
    validate(Accessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE_END() // CpuPool2d
TEST_SUITE_END() // OPERATORS
TEST_SUITE_END() // NEON

} // namespace validation
} // namespace test
} // namespace arm_compute
