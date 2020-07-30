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
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCFullyConnectedLayer.h"
#include "tests/GLES_COMPUTE/GCAccessor.h"
#include "tests/GLES_COMPUTE/Helper.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/FullyConnectedLayerDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/FullyConnectedLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Tolerance for float operations */
constexpr RelativeTolerance<float>  rel_tolerance_f32(0.05f);
constexpr AbsoluteTolerance<float>  abs_tolerance_f32(0.0001f);
RelativeTolerance<half_float::half> tolerance_f16(half(0.2));
constexpr float                     tolerance_num = 0.07f; /**< Tolerance number */

/** CNN data types */
const auto CNNDataTypes = framework::dataset::make("DataType",
{
    DataType::F16,
    DataType::F32,
});

const auto FullyConnectedParameters   = combine(framework::dataset::make("TransposeWeights", { false, true }), framework::dataset::make("ReshapeWeights", { false, true }));
const auto ActivationFunctionsDataset = framework::dataset::make("ActivationInfo", ActivationLayerInfo());
} // namespace

TEST_SUITE(GC)
TEST_SUITE(FullyConnectedLayer)

template <typename T>
using GCFullyConnectedLayerFixture = FullyConnectedLayerValidationFixture<GCTensor, GCAccessor, GCFullyConnectedLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, GCFullyConnectedLayerFixture<half_float::half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallFullyConnectedLayerDataset(),
                       FullyConnectedParameters),
                       framework::dataset::make("DataType", DataType::F16)),
                       ActivationFunctionsDataset))
{
    // Validate output
    validate(GCAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
FIXTURE_DATA_TEST_CASE(RunLarge, GCFullyConnectedLayerFixture<half_float::half>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeFullyConnectedLayerDataset(),
                       FullyConnectedParameters),
                       framework::dataset::make("DataType", DataType::F16)),
                       ActivationFunctionsDataset))
{
    // Validate output
    validate(GCAccessor(_target), _reference, tolerance_f16, tolerance_num);
}
TEST_SUITE_END()

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, GCFullyConnectedLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(datasets::SmallFullyConnectedLayerDataset(), FullyConnectedParameters),
                                                                                                                 framework::dataset::make("DataType", DataType::F32)),
                                                                                                                 ActivationFunctionsDataset))
{
    // Validate output
    validate(GCAccessor(_target), _reference, rel_tolerance_f32);
}
FIXTURE_DATA_TEST_CASE(RunLarge, GCFullyConnectedLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeFullyConnectedLayerDataset(), FullyConnectedParameters),
                                                                                                                       framework::dataset::make("DataType", DataType::F32)),
                                                                                                               ActivationFunctionsDataset))
{
    // Validate output
    validate(GCAccessor(_target), _reference, rel_tolerance_f32, 0, abs_tolerance_f32);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
