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
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLPoolingLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/PoolingLayerDataset.h"
#include "tests/datasets/PoolingTypesDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/PoolingLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Input data sets for floating-point data types */
const auto PoolingLayerDatasetFP = combine(combine(combine(datasets::PoolingTypes(), framework::dataset::make("PoolingSize", { Size2D(2, 2), Size2D(3, 3), Size2D(5, 7) })),
                                                   framework::dataset::make("PadStride", { PadStrideInfo(1, 1, 0, 0), PadStrideInfo(2, 1, 0, 0), PadStrideInfo(1, 2, 1, 1), PadStrideInfo(2, 2, 1, 0) })),
                                           framework::dataset::make("ExcludePadding", { true, false }));

const auto PoolingLayerDatasetFPSmall = combine(combine(combine(datasets::PoolingTypes(), framework::dataset::make("PoolingSize", { Size2D(2, 2), Size2D(3, 3) })),
                                                        framework::dataset::make("PadStride", { PadStrideInfo(1, 1, 0, 0), PadStrideInfo(2, 1, 0, 0) })),
                                                framework::dataset::make("ExcludePadding", { true, false }));

/** Input data sets for asymmetric data type */
const auto PoolingLayerDatasetQASYMM8 = combine(concat(combine(combine(framework::dataset::make("PoolingType",
{
    PoolingType::MAX, PoolingType::AVG,
}),
framework::dataset::make("PoolingSize", { Size2D(2, 2), Size2D(3, 3) })),
framework::dataset::make("PadStride", { PadStrideInfo(1, 1, 0, 0), PadStrideInfo(1, 2, 1, 1), PadStrideInfo(2, 2, 1, 0) })),
combine(combine(framework::dataset::make("PoolingType", { PoolingType::AVG }), framework::dataset::make("PoolingSize", { Size2D(5, 7) })), framework::dataset::make("PadStride", { PadStrideInfo(2, 1, 0, 0) }))),
framework::dataset::make("ExcludePadding", { true }));

const auto PoolingLayerDatasetQASYMM8Small = combine(combine(combine(framework::dataset::make("PoolingType",
{
    PoolingType::MAX, PoolingType::AVG,
}),
framework::dataset::make("PoolingSize", { Size2D(2, 2), Size2D(5, 7) })),
framework::dataset::make("PadStride", { PadStrideInfo(1, 2, 1, 1) })),
framework::dataset::make("ExcludePadding", { true }));

constexpr AbsoluteTolerance<float>   tolerance_f32(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for 32-bit floating-point type */
constexpr AbsoluteTolerance<float>   tolerance_f16(0.01f);  /**< Tolerance value for comparing reference's output against implementation's output for 16-bit floating-point type */
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(1);  /**< Tolerance value for comparing reference's output against implementation's output for 8-bit asymmetric type */
const auto                           pool_data_layout_dataset = framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC });

} // namespace

TEST_SUITE(CL)
TEST_SUITE(PoolingLayer)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Mismatching data type
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Window shrink
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Invalid pad/size combination
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Invalid pad/size combination
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::QASYMM8), // Invalid parameters
                                                       TensorInfo(TensorShape(15U, 13U, 5U), 1, DataType::F32),     // Non-rectangular Global Pooling
                                                       TensorInfo(TensorShape(13U, 13U, 5U), 1, DataType::F32),     // Invalid output Global Pooling
                                                       TensorInfo(TensorShape(13U, 13U, 5U), 1, DataType::F32),
                                                     }),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(30U, 11U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(25U, 16U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::QASYMM8),
                                                       TensorInfo(TensorShape(1U, 1U, 5U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(2U, 2U, 5U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(1U, 1U, 5U), 1, DataType::F32),
                                                     })),
               framework::dataset::make("PoolInfo",  { PoolingLayerInfo(PoolingType::AVG, 3, PadStrideInfo(1, 1, 0, 0)),
                                                       PoolingLayerInfo(PoolingType::AVG, 3, PadStrideInfo(1, 1, 0, 0)),
                                                       PoolingLayerInfo(PoolingType::AVG, 2, PadStrideInfo(1, 1, 2, 0)),
                                                       PoolingLayerInfo(PoolingType::AVG, 2, PadStrideInfo(1, 1, 0, 2)),
                                                       PoolingLayerInfo(PoolingType::L2, 3, PadStrideInfo(1, 1, 0, 0)),
                                                       PoolingLayerInfo(PoolingType::AVG),
                                                       PoolingLayerInfo(PoolingType::MAX),
                                                       PoolingLayerInfo(PoolingType::AVG),
                                                      })),
               framework::dataset::make("Expected", { false, false, false, false, false, true, false, true })),
               input_info, output_info, pool_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLPoolingLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), pool_info)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLPoolingLayerFixture = PoolingLayerValidationFixture<CLTensor, CLAccessor, CLPoolingLayer, T>;

template <typename T>
using CLSpecialPoolingLayerFixture = SpecialPoolingLayerValidationFixture<CLTensor, CLAccessor, CLPoolingLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSpecial, CLSpecialPoolingLayerFixture<float>, framework::DatasetMode::ALL, datasets::PoolingLayerDatasetSpecial() * framework::dataset::make("DataType", DataType::F32))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunSmall, CLPoolingLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), combine(PoolingLayerDatasetFPSmall,
                                                                                                                  framework::dataset::make("DataType",
                                                                                                                          DataType::F32))),
                                                                                                          pool_data_layout_dataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLPoolingLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), combine(PoolingLayerDatasetFP,
                                                                                                                framework::dataset::make("DataType",
                                                                                                                        DataType::F32))),
                                                                                                        pool_data_layout_dataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // FP32

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLPoolingLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), combine(PoolingLayerDatasetFPSmall,
                                                                                                                 framework::dataset::make("DataType", DataType::F16))),
                                                                                                         pool_data_layout_dataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLPoolingLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), combine(PoolingLayerDatasetFP,
                                                                                                               framework::dataset::make("DataType", DataType::F16))),
                                                                                                       pool_data_layout_dataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END() // FP16
TEST_SUITE_END() // Float

TEST_SUITE(Quantized)

template <typename T>
using CLPoolingLayerQuantizedFixture = PoolingLayerValidationQuantizedFixture<CLTensor, CLAccessor, CLPoolingLayer, T>;

TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLPoolingLayerQuantizedFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), combine(PoolingLayerDatasetQASYMM8Small,
                                                                                                                     framework::dataset::make("DataType", DataType::QASYMM8))),
                                                                                                                     pool_data_layout_dataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLPoolingLayerQuantizedFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), combine(PoolingLayerDatasetQASYMM8,
                                                                                                                   framework::dataset::make("DataType", DataType::QASYMM8))),
                                                                                                                   pool_data_layout_dataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qasymm8);
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE_END() // Quantized
TEST_SUITE_END() // PoolingLayer
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
