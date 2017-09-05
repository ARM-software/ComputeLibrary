/*
 * Copyright (c) 2017 ARM Limited.
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
/** Input data set for float data types */
const auto PoolingLayerDatasetFP = combine(combine(datasets::PoolingTypes(), framework::dataset::make("PoolingSize", { 2, 3, 7, 9 })),
                                           framework::dataset::make("PadStride", { PadStrideInfo(1, 1, 0, 0), PadStrideInfo(2, 1, 0, 0), PadStrideInfo(1, 2, 1, 1), PadStrideInfo(2, 2, 1, 0) }));

/** Input data set for quantized data types */
const auto PoolingLayerDatasetQS = combine(combine(datasets::PoolingTypes(), framework::dataset::make("PoolingSize", { 2, 3 })),
                                           framework::dataset::make("PadStride", { PadStrideInfo(1, 1, 0, 0), PadStrideInfo(2, 1, 0, 0), PadStrideInfo(1, 2, 1, 1), PadStrideInfo(2, 2, 1, 0) }));

constexpr AbsoluteTolerance<float> tolerance_f32(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for float types */
constexpr AbsoluteTolerance<float> tolerance_f16(0.01f);  /**< Tolerance value for comparing reference's output against implementation's output for float types */
constexpr AbsoluteTolerance<float> tolerance_qs8(3);      /**< Tolerance value for comparing reference's output against implementation's output for quantized input */
constexpr AbsoluteTolerance<float> tolerance_qs16(6);     /**< Tolerance value for comparing reference's output against implementation's output for quantized input */
} // namespace

TEST_SUITE(CL)
TEST_SUITE(PoolingLayer)

template <typename T>
using CLPoolingLayerFixture = PoolingLayerValidationFixture<CLTensor, CLAccessor, CLPoolingLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLPoolingLayerFixture<float>, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), combine(PoolingLayerDatasetFP, framework::dataset::make("DataType",
                                                                                                    DataType::F32))))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLPoolingLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), combine(PoolingLayerDatasetFP, framework::dataset::make("DataType",
                                                                                                        DataType::F32))))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END()

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLPoolingLayerFixture<half>, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), combine(PoolingLayerDatasetFP,
                                                                                                   framework::dataset::make("DataType", DataType::F16))))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLPoolingLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), combine(PoolingLayerDatasetFP,
                                                                                                       framework::dataset::make("DataType", DataType::F16))))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using CLPoolingLayerFixedPointFixture = PoolingLayerValidationFixedPointFixture<CLTensor, CLAccessor, CLPoolingLayer, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QS8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLPoolingLayerFixedPointFixture<int8_t>, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), combine(PoolingLayerDatasetQS,
                                                                                                                       framework::dataset::make("DataType", DataType::QS8))),
                                                                                                               framework::dataset::make("FractionalBits", 1, 4)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qs8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLPoolingLayerFixedPointFixture<int8_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), combine(PoolingLayerDatasetQS,
                                                                                                                   framework::dataset::make("DataType", DataType::QS8))),
                                                                                                                   framework::dataset::make("FractionalBits", 1, 4)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qs8);
}
TEST_SUITE_END()

TEST_SUITE(QS16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLPoolingLayerFixedPointFixture<int16_t>, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), combine(PoolingLayerDatasetQS,
                                                                                                                        framework::dataset::make("DataType", DataType::QS16))),
                                                                                                                framework::dataset::make("FractionalBits", 1, 12)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qs16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLPoolingLayerFixedPointFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), combine(PoolingLayerDatasetQS,
                                                                                                                    framework::dataset::make("DataType", DataType::QS16))),
                                                                                                                    framework::dataset::make("FractionalBits", 1, 12)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qs16);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
