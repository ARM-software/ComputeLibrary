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
#include "arm_compute/runtime/CL/functions/CLNormalizationLayer.h"
#include "framework/Asserts.h"
#include "framework/Macros.h"
#include "framework/datasets/Datasets.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets_new/NormalizationTypesDataset.h"
#include "tests/datasets_new/ShapeDatasets.h"
#include "tests/validation_new/Validation.h"
#include "tests/validation_new/fixtures/NormalizationLayerFixture.h"
#include "tests/validation_new/half.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Tolerance for float operations */
constexpr AbsoluteTolerance<float> tolerance_f16(0.01f);
constexpr AbsoluteTolerance<float> tolerance_f32(0.00001f);
/** Tolerance for fixed point operations */
constexpr AbsoluteTolerance<int8_t>  tolerance_qs8(2);
constexpr AbsoluteTolerance<int16_t> tolerance_qs16(2);

/** Input data set. */
const auto NormalizationDataset = combine(combine(combine(datasets::SmallShapes(), framework::dataset::make("NormType", { NormType::IN_MAP_1D, NormType::CROSS_MAP })),
                                                  framework::dataset::make("NormalizationSize", 3, 9, 2)),
                                          framework::dataset::make("Beta", { 0.5f, 1.f, 2.f }));
} // namespace

TEST_SUITE(CL)
TEST_SUITE(NormalizationLayer)

//TODO(COMPMID-415): Missing configuration?

template <typename T>
using CLNormalizationLayerFixture = NormalizationValidationFixture<CLTensor, CLAccessor, CLNormalizationLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLNormalizationLayerFixture<half_float::half>, framework::DatasetMode::PRECOMMIT, combine(NormalizationDataset, framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLNormalizationLayerFixture<half_float::half>, framework::DatasetMode::NIGHTLY, combine(NormalizationDataset, framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END()

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLNormalizationLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(NormalizationDataset, framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLNormalizationLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(NormalizationDataset, framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END()
TEST_SUITE_END()

template <typename T>
using CLNormalizationLayerFixedPointFixture = NormalizationValidationFixedPointFixture<CLTensor, CLAccessor, CLNormalizationLayer, T>;

TEST_SUITE(Quantized)
TEST_SUITE(QS8)
// Testing for fixed point position [1,6) as reciprocal limits the maximum fixed point position to 5
FIXTURE_DATA_TEST_CASE(RunSmall, CLNormalizationLayerFixedPointFixture<int8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(NormalizationDataset, framework::dataset::make("DataType",
                       DataType::QS8)),
                       framework::dataset::make("FractionalBits", 1, 6)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qs8);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLNormalizationLayerFixedPointFixture<int8_t>, framework::DatasetMode::NIGHTLY, combine(combine(NormalizationDataset, framework::dataset::make("DataType",
                       DataType::QS8)),
                       framework::dataset::make("FractionalBits", 1, 6)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qs8);
}
TEST_SUITE_END()

TEST_SUITE(QS16)
// Testing for fixed point position [1,14) as reciprocal limits the maximum fixed point position to 5
FIXTURE_DATA_TEST_CASE(RunSmall, CLNormalizationLayerFixedPointFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(NormalizationDataset, framework::dataset::make("DataType",
                       DataType::QS16)),
                       framework::dataset::make("FractionalBits", 1, 14)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance_qs16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, CLNormalizationLayerFixedPointFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(NormalizationDataset, framework::dataset::make("DataType",
                       DataType::QS16)),
                       framework::dataset::make("FractionalBits", 1, 14)))
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
