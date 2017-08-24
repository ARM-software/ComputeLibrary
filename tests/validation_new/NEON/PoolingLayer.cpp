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
#include "arm_compute/runtime/NEON/functions/NEPoolingLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "framework/Asserts.h"
#include "framework/Macros.h"
#include "framework/datasets/Datasets.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets_new/PoolingTypesDataset.h"
#include "tests/datasets_new/ShapeDatasets.h"
#include "tests/validation_new/Validation.h"
#include "tests/validation_new/fixtures/PoolingLayerFixture.h"
#include "tests/validation_new/half.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Input data set for float data types */
const auto PoolingLayerDatasetFP = combine(combine(datasets::PoolingTypes(), framework::dataset::make("PoolingSize", { 2, 3, 7 })),
                                           framework::dataset::make("PadStride", { PadStrideInfo(1, 1, 0, 0), PadStrideInfo(2, 1, 0, 0), PadStrideInfo(1, 2, 1, 1), PadStrideInfo(2, 2, 1, 0) }));

/** Input data set for quantized data types */
const auto PoolingLayerDatasetQS = combine(combine(datasets::PoolingTypes(), framework::dataset::make("PoolingSize", { 2, 3 })),
                                           framework::dataset::make("PadStride", { PadStrideInfo(1, 1, 0, 0), PadStrideInfo(2, 1, 0, 0), PadStrideInfo(1, 2, 1, 1), PadStrideInfo(2, 2, 1, 0) }));

constexpr AbsoluteTolerance<float> tolerance_f32(0.001f); /**< Tolerance value for comparing reference's output against implementation's output for float types */
#ifdef ARM_COMPUTE_ENABLE_FP16
constexpr AbsoluteTolerance<float> tolerance_f16(0.01f); /**< Tolerance value for comparing reference's output against implementation's output for float types */
#endif                                                   /* ARM_COMPUTE_ENABLE_FP16 */
constexpr AbsoluteTolerance<float> tolerance_qs8(0);     /**< Tolerance value for comparing reference's output against implementation's output for quantized input */
constexpr AbsoluteTolerance<float> tolerance_qs16(0);    /**< Tolerance value for comparing reference's output against implementation's output for quantized input */
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(PoolingLayer)

//TODO(COMPMID-415): Configuration tests?

template <typename T>
using NEPoolingLayerFixture = PoolingLayerValidationFixture<Tensor, Accessor, NEPoolingLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEPoolingLayerFixture<float>, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), combine(PoolingLayerDatasetFP, framework::dataset::make("DataType",
                                                                                                    DataType::F32))))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEPoolingLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), combine(PoolingLayerDatasetFP, framework::dataset::make("DataType",
                                                                                                        DataType::F32))))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END()

#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEPoolingLayerFixture<half_float::half>, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), combine(PoolingLayerDatasetFP,
                                                                                                               framework::dataset::make("DataType", DataType::F16))))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEPoolingLayerFixture<half_float::half>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), combine(PoolingLayerDatasetFP,
                                                                                                                   framework::dataset::make("DataType", DataType::F16))))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END()
#endif /* ARM_COMPUTE_ENABLE_FP16 */
TEST_SUITE_END()

template <typename T>
using NEPoolingLayerFixedPointFixture = PoolingLayerValidationFixedPointFixture<Tensor, Accessor, NEPoolingLayer, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QS8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEPoolingLayerFixedPointFixture<int8_t>, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), combine(PoolingLayerDatasetQS,
                                                                                                                       framework::dataset::make("DataType", DataType::QS8))),
                                                                                                               framework::dataset::make("FractionalBits", 1, 5)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qs8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEPoolingLayerFixedPointFixture<int8_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), combine(PoolingLayerDatasetQS,
                                                                                                                   framework::dataset::make("DataType", DataType::QS8))),
                                                                                                                   framework::dataset::make("FractionalBits", 1, 5)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qs8);
}
TEST_SUITE_END()

TEST_SUITE(QS16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEPoolingLayerFixedPointFixture<int16_t>, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), combine(PoolingLayerDatasetQS,
                                                                                                                        framework::dataset::make("DataType", DataType::QS16))),
                                                                                                                framework::dataset::make("FractionalBits", 1, 13)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qs16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEPoolingLayerFixedPointFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), combine(PoolingLayerDatasetQS,
                                                                                                                    framework::dataset::make("DataType", DataType::QS16))),
                                                                                                                    framework::dataset::make("FractionalBits", 1, 13)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_qs16);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
