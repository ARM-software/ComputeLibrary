/*
 * Copyright (c) 2024 Arm Limited.
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
#include "arm_compute/runtime/experimental/operators/CpuActivation.h"
#include "tests/NEON/Accessor.h"
#include "tests/datasets/ActivationFunctionsDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/CpuActivationFixture.h"

/*
 * Tests for arm_compute::experimental::op::CpuActivation which is a shallow wrapper for
 * arm_compute::cpu::CpuActivation. Any future testing to the functionalities of cpu::CpuActivation
 * will be tested in tests/NEON/ActivationLayer.cpp given that op::CpuActivation remain a
 * shallow wrapper.
*/

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Define relative tolerance of the activation layer.
 *
 * @param[in] activation The activation function used.
 *
 * @return Relative tolerance depending on the activation function.
 */
RelativeTolerance<float> relative_tolerance(ActivationLayerInfo::ActivationFunction activation)
{
    switch(activation)
    {
        case ActivationLayerInfo::ActivationFunction::LOGISTIC:
        case ActivationLayerInfo::ActivationFunction::ELU:
        case ActivationLayerInfo::ActivationFunction::SQRT:
        case ActivationLayerInfo::ActivationFunction::TANH:
        case ActivationLayerInfo::ActivationFunction::HARD_SWISH:
        case ActivationLayerInfo::ActivationFunction::SWISH:
        case ActivationLayerInfo::ActivationFunction::GELU:
            return RelativeTolerance<float>(0.05f);
        case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
            return RelativeTolerance<float>(0.00001f);
        default:
            return RelativeTolerance<float>(0.f);
    }
}

/** Define absolute tolerance of the activation layer.
 *
 * @param[in] activation The activation function used.
 *
 * @return Absolute tolerance depending on the activation function.
 */
AbsoluteTolerance<float> absolute_tolerance(ActivationLayerInfo::ActivationFunction activation)
{
    switch(activation)
    {
        case ActivationLayerInfo::ActivationFunction::LOGISTIC:
        case ActivationLayerInfo::ActivationFunction::SQRT:
        case ActivationLayerInfo::ActivationFunction::TANH:
        case ActivationLayerInfo::ActivationFunction::SWISH:
        case ActivationLayerInfo::ActivationFunction::HARD_SWISH:
        case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
            return AbsoluteTolerance<float>(0.00001f);
        default:
            return AbsoluteTolerance<float>(0.f);
    }
}

const auto NeonActivationFunctionsDataset = concat(datasets::ActivationFunctions(),
                                                   framework::dataset::make("ActivationFunction", { ActivationLayerInfo::ActivationFunction::HARD_SWISH, ActivationLayerInfo::ActivationFunction::SWISH }));

/** Input data sets. */
const auto ActivationDataset = combine(combine(framework::dataset::make("InPlace", { false, true }), NeonActivationFunctionsDataset), framework::dataset::make("AlphaBeta", { 0.5f, 1.f }));

} // namespace

TEST_SUITE(NEON)
TEST_SUITE(OPERATORS)
TEST_SUITE(CpuActivation)

template <typename T>
using CpuActivationFixture = CpuActivationValidationFixture<Tensor, Accessor, experimental::op::CpuActivation, T>;

TEST_SUITE(SmokeTest)
FIXTURE_DATA_TEST_CASE(SmokeTest, CpuActivationFixture<float>, framework::DatasetMode::ALL, combine(combine(datasets::SmallShapes(), ActivationDataset), framework::dataset::make("DataType",
                                                                                                       DataType::F32)))

{
    // Validate output
    validate(Accessor(_target), _reference, relative_tolerance(_function), 0.f, absolute_tolerance(_function));
}
TEST_SUITE_END() // SmokeTest

TEST_SUITE_END() // CpuActivation
TEST_SUITE_END() // OPERATORS
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
