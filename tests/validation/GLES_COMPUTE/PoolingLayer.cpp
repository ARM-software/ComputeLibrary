/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/runtime/GLES_COMPUTE/GCTensor.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCTensorAllocator.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCPoolingLayer.h"
#include "tests/GLES_COMPUTE/GCAccessor.h"
#include "tests/PaddingCalculator.h"
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
/** Input data set for floating-point data types */
const auto PoolingLayerDatasetFP = combine(combine(combine(datasets::PoolingTypes(), framework::dataset::make("PoolingSize", { Size2D(2, 2), Size2D(3, 3), Size2D(4, 4), Size2D(7, 7), Size2D(9, 9) })),
                                                   framework::dataset::make("PadStride", { PadStrideInfo(1, 1, 0, 0), PadStrideInfo(2, 1, 0, 0), PadStrideInfo(1, 2, 1, 1), PadStrideInfo(2, 2, 1, 0) })),
                                           framework::dataset::make("ExcludePadding", { true, false }));

constexpr AbsoluteTolerance<float> tolerance_f32(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for 32-bit floating-point type */
constexpr AbsoluteTolerance<float> tolerance_f16(0.01f);  /**< Tolerance value for comparing reference's output against implementation's output for 16-bit floating-point type */
} // namespace

TEST_SUITE(GC)
TEST_SUITE(PoolingLayer)

//clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
                                                                  framework::dataset::make("InputInfo",
{
    TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Mismatching data type
    TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Window shrink
    TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Invalid pad/size combination
    TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32), // Invalid pad/size combination
    TensorInfo(TensorShape(15U, 13U, 5U), 1, DataType::F32), // Non-rectangular Global Pooling
    TensorInfo(TensorShape(13U, 13U, 5U), 1, DataType::F32), // Invalid output Global Pooling
    TensorInfo(TensorShape(13U, 13U, 5U), 1, DataType::F32),
}),
framework::dataset::make("OutputInfo",
{
    TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F16), TensorInfo(TensorShape(25U, 11U, 2U), 1, DataType::F32), TensorInfo(TensorShape(30U, 11U, 2U), 1, DataType::F32), TensorInfo(TensorShape(25U, 16U, 2U), 1, DataType::F32), TensorInfo(TensorShape(1U, 1U, 5U), 1, DataType::F32), TensorInfo(TensorShape(2U, 2U, 5U), 1, DataType::F32), TensorInfo(TensorShape(1U, 1U, 5U), 1, DataType::F32),
})),
framework::dataset::make("PoolInfo",
{
    PoolingLayerInfo(PoolingType::AVG, 3, DataLayout::NCHW, PadStrideInfo(1, 1, 0, 0)),
    PoolingLayerInfo(PoolingType::AVG, 3, DataLayout::NCHW, PadStrideInfo(1, 1, 0, 0)),
    PoolingLayerInfo(PoolingType::AVG, 3, DataLayout::NCHW, PadStrideInfo(1, 1, 0, 0)),
    PoolingLayerInfo(PoolingType::AVG, 3, DataLayout::NCHW, PadStrideInfo(1, 1, 0, 0)),
    PoolingLayerInfo(PoolingType::AVG, 2, DataLayout::NCHW, PadStrideInfo(1, 1, 2, 0)),
    PoolingLayerInfo(PoolingType::AVG, 2, DataLayout::NCHW, PadStrideInfo(1, 1, 0, 2)),
    PoolingLayerInfo(PoolingType::L2, 3, DataLayout::NCHW, PadStrideInfo(1, 1, 0, 0)),
    PoolingLayerInfo(PoolingType::AVG, DataLayout::NCHW),
    PoolingLayerInfo(PoolingType::MAX, DataLayout::NCHW),
    PoolingLayerInfo(PoolingType::AVG, DataLayout::NCHW),
})),
framework::dataset::make("Expected", { false, false, false, false, false, false, false, false, false, true })),
input_info, output_info, pool_info, expected)
{
    ARM_COMPUTE_EXPECT(bool(GCPoolingLayer::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), pool_info)) == expected, framework::LogLevel::ERRORS);
}
//clang-format on

template <typename T>
using GCPoolingLayerFixture = PoolingLayerValidationFixture<GCTensor, GCAccessor, GCPoolingLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, GCPoolingLayerFixture<float>, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), combine(PoolingLayerDatasetFP, framework::dataset::make("DataType",
                                                                                                            DataType::F32))),
                                                                                                    framework::dataset::make("DataLayout", DataLayout::NCHW)))
{
    // Validate output
    validate(GCAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, GCPoolingLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), combine(PoolingLayerDatasetFP,
                                                                                                                framework::dataset::make("DataType",
                                                                                                                        DataType::F32))),
                                                                                                        framework::dataset::make("DataLayout", DataLayout::NCHW)))
{
    // Validate output
    validate(GCAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END()

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, GCPoolingLayerFixture<half>, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), combine(PoolingLayerDatasetFP,
                                                                                                           framework::dataset::make("DataType", DataType::F16))),
                                                                                                   framework::dataset::make("DataLayout", DataLayout::NCHW)))
{
    // Validate output
    validate(GCAccessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, GCPoolingLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), combine(PoolingLayerDatasetFP,
                                                                                                               framework::dataset::make("DataType", DataType::F16))),
                                                                                                       framework::dataset::make("DataLayout", DataLayout::NCHW)))
{
    // Validate output
    validate(GCAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
