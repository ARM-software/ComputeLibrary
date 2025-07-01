/*
 * Copyright (c) 2017-2021, 2023-2025 Arm Limited.
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

// This experimental stateless wrapper only supports fixed-format weights.
#ifdef ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS

#include "arm_compute/runtime/experimental/operators/CpuFullyConnected.h"
#include "arm_compute/function_info/ActivationLayerInfo.h"

#include "tests/datasets/FullyConnectedLayerDataset.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/NEON/Accessor.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/CpuFullyConnectedFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
using framework::dataset::make;
namespace
{
/** Tolerance for float operations */
constexpr RelativeTolerance<float> tolerance_f32(0.01f);  /**< Relative tolerance value for comparing reference's output against implementation's output for DataType::F32 */
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(OPERATORS)
TEST_SUITE(CpuFullyConnected)

template <typename T>
using CpuFullyConnectedFixture = CpuFullyConnectedValidationFixture<Tensor, Accessor, experimental::op::CpuFullyConnected, T>;

template <typename T>
using CpuFullyConnectedThreadSafeFixture = CpuFullyConnectedThreadSafeValidationFixture<Tensor, Accessor, experimental::op::CpuFullyConnected, T>;

TEST_SUITE(SmokeTest)
// We only test FP32 here because we do not currently have reorder support for
// FP16 tensors. However, the behaviour is expected to be very similar.
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(SmokeTest,
                       CpuFullyConnectedFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallFullyConnectedLayerDataset(),
                               make("DataType", DataType::F32),
                               make("ActivationInfo", { ActivationLayerInfo(), ActivationLayerInfo(ActivationFunction::LOGISTIC) })))
{
    // Validate output
    for(int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i], tolerance_f32);
    }
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // SmokeTest

// FP16 tests are missing due to lack of reorder support (see note above).
// CpuActivation is already covered by its own thread-safety tests.
TEST_SUITE(ThreadSafety)
FIXTURE_DATA_TEST_CASE(ConfigureOnceUseFromDifferentThreads,
                       CpuFullyConnectedThreadSafeFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallFullyConnectedLayerDataset(),
                               make("DataType", DataType::F32),
                               make("ActivationInfo", { ActivationLayerInfo(), ActivationLayerInfo(ActivationFunction::LOGISTIC) })))
{
    // Validate output
    for(int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i], tolerance_f32);
    }
}
TEST_SUITE_END() // ThreadSafety

TEST_SUITE_END() // CpuFullyConnected
TEST_SUITE_END() // OPERATORS
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
