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
#include "arm_compute/runtime/Allocator.h"
#include "arm_compute/runtime/NEON/functions/NEFlattenLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/FlattenLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(NEON)
TEST_SUITE(FlattenLayer)

template <typename T>
using NEFlattenLayerFixture = FlattenLayerValidationFixture<Tensor, Accessor, NEFlattenLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEFlattenLayerFixture<float>, framework::DatasetMode::ALL, combine(framework::dataset::concat(datasets::Small3DShapes(), datasets::Small4DShapes()),
                                                                                                    framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEFlattenLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(framework::dataset::concat(datasets::Large3DShapes(), datasets::Large4DShapes()),
                                                                                                        framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEFlattenLayerFixture<half>, framework::DatasetMode::ALL, combine(framework::dataset::concat(datasets::Small3DShapes(), datasets::Small4DShapes()),
                                                                                                   framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEFlattenLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(framework::dataset::concat(datasets::Large3DShapes(), datasets::Large4DShapes()),
                                                                                                       framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
TEST_SUITE_END()

TEST_SUITE(Quantized)
TEST_SUITE(QS8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEFlattenLayerFixture<int8_t>, framework::DatasetMode::ALL, combine(framework::dataset::concat(datasets::Small3DShapes(), datasets::Small4DShapes()),
                                                                                                     framework::dataset::make("DataType", DataType::QS8)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEFlattenLayerFixture<int8_t>, framework::DatasetMode::NIGHTLY, combine(framework::dataset::concat(datasets::Large3DShapes(), datasets::Large4DShapes()),
                                                                                                         framework::dataset::make("DataType", DataType::QS8)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE(QS16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEFlattenLayerFixture<int16_t>, framework::DatasetMode::ALL, combine(framework::dataset::concat(datasets::Small3DShapes(), datasets::Small4DShapes()),
                                                                                                      framework::dataset::make("DataType", DataType::QS16)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEFlattenLayerFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(framework::dataset::concat(datasets::Large3DShapes(), datasets::Large4DShapes()),
                                                                                                          framework::dataset::make("DataType", DataType::QS16)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
