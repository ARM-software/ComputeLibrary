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
#include "arm_compute/runtime/NEON/functions/NEYOLOLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ActivationFunctionsDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/YOLOLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Tolerance */
constexpr AbsoluteTolerance<float> tolerance_f32(1e-6f);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
constexpr RelativeTolerance<float> tolerance_f16(0.001f);
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

/** Floating point data sets. */
const auto YOLODataset = combine(combine(combine(combine(framework::dataset::make("InPlace", { false, true }), framework::dataset::make("ActivationFunction",
                                                         ActivationLayerInfo::ActivationFunction::LOGISTIC)),
                                                 framework::dataset::make("AlphaBeta", { 0.5f, 1.f })),
                                         framework::dataset::make("Classes", 40)),
                                 framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC }));
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(YOLOLayer)

template <typename T>
using NEYOLOLayerFixture = YOLOValidationFixture<Tensor, Accessor, NEYOLOLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEYOLOLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallYOLOShapes(), YOLODataset), framework::dataset::make("DataType",
                                                                                                       DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEYOLOLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeYOLOShapes(), YOLODataset), framework::dataset::make("DataType",
                                                                                                     DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f32);
}
TEST_SUITE_END() // FP32

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEYOLOLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallYOLOShapes(), YOLODataset), framework::dataset::make("DataType",
                                                                                                      DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NEYOLOLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeYOLOShapes(), YOLODataset), framework::dataset::make("DataType",
                                                                                                    DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_f16);
}
TEST_SUITE_END() // FP16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
TEST_SUITE_END() // Float

TEST_SUITE_END() // YOLOLayer
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
