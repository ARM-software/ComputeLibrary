/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEConvertFullyConnectedWeights.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ConvertFullyConnectedWeightsFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
auto params = combine(framework::dataset::make("WeightsWidth", { 16, 32, 64 }), framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC }));
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(ConvertFullyConnectedWeights)

template <typename T>
using NEConvertFullyConnectedWeightsFixture = ConvertFullyConnectedWeightsValidationFixture<Tensor, Accessor, NEConvertFullyConnectedWeights, T>;

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEConvertFullyConnectedWeightsFixture<float>, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), combine(params, framework::dataset::make("DataType",
                                                                                                                    DataType::F32))))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEConvertFullyConnectedWeightsFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), combine(params, framework::dataset::make("DataType",
                                                                                                                        DataType::F32))))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEConvertFullyConnectedWeightsFixture<half>, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), combine(params, framework::dataset::make("DataType",
                                                                                                                   DataType::F16))))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEConvertFullyConnectedWeightsFixture<half>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), combine(params, framework::dataset::make("DataType",
                                                                                                                       DataType::F16))))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, NEConvertFullyConnectedWeightsFixture<uint8_t>, framework::DatasetMode::ALL, combine(datasets::SmallShapes(), combine(params, framework::dataset::make("DataType",
                                                                                                                      DataType::QASYMM8))))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEConvertFullyConnectedWeightsFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeShapes(), combine(params, framework::dataset::make("DataType",
                       DataType::QASYMM8))))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
