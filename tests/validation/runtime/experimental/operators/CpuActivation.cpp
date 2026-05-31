/*
 * Copyright (c) 2024-2026 Arm Limited.
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

#include "tests/datasets/ActivationFunctionsDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/validation/fixtures/CpuActivationFixture.h"
#include "tests/validation/helpers/ActivationHelpers.h"
#include "tests/validation/Validation.h"

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
using framework::dataset::make;
namespace
{

const auto NeonActivationFunctionsDataset =
    concat(datasets::ActivationFunctions(),
           make("ActivationFunction",
                {ActivationLayerInfo::ActivationFunction::HARD_SWISH, ActivationLayerInfo::ActivationFunction::SWISH}));

/** Input data sets. */
const auto ActivationDataset =
    combine(make("InPlace", {false, true}), NeonActivationFunctionsDataset, make("AlphaBeta", {0.5f, 1.f}));

// Inplace calculation is irrelevant to thread safety because different threads
// will use different tensors. AlphaBeta value is also irrelevant as it's just
// a change in the computation value.
const auto FloatActivationDatasetForThreadSafetyTests =
    combine(make("InPlace", {false}), NeonActivationFunctionsDataset, make("AlphaBeta", {0.5f}));

const auto QuantizedActivationFunctionsDataset = make("ActivationFunction",
                                                      {
                                                          ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
                                                          ActivationLayerInfo::ActivationFunction::RELU,
                                                          ActivationLayerInfo::ActivationFunction::BOUNDED_RELU,
                                                          ActivationLayerInfo::ActivationFunction::LOGISTIC,
                                                          ActivationLayerInfo::ActivationFunction::TANH,
                                                          ActivationLayerInfo::ActivationFunction::LEAKY_RELU,
                                                      });

const auto QuantizedActivationDatasetForThreadSafetyTests =
    combine(make("InPlace", {false}),
            concat(QuantizedActivationFunctionsDataset,
                   make("ActivationFunction", ActivationLayerInfo::ActivationFunction::HARD_SWISH)),
            make("AlphaBeta", {1.f}));
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(OPERATORS)
TEST_SUITE(CpuActivation)

template <typename T>
using CpuActivationFixture = CpuActivationValidationFixture<Tensor, Accessor, experimental::op::CpuActivation, T>;

template <typename T>
using CpuActivationFloatThreadSafeFixture =
    CpuActivationFloatThreadSafeValidationFixture<Tensor, Accessor, experimental::op::CpuActivation, T>;

template <typename T>
using CpuActivationQuantizedThreadSafeFixture =
    CpuActivationQuantizedThreadSafeValidationFixture<Tensor, Accessor, experimental::op::CpuActivation, T>;

TEST_SUITE(SmokeTest)
FIXTURE_DATA_TEST_CASE(SmokeTest,
                       CpuActivationFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(), ActivationDataset, make("DataType", DataType::F32)))
{
    // Validate output
    for (int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i], helper::relative_tolerance(_data_type, _function),
                 helper::tolerance_num(_data_type, _function), helper::absolute_tolerance(_data_type, _function));
    }
}
TEST_SUITE_END() // SmokeTest

#ifndef BARE_METAL
TEST_SUITE(ThreadSafety)
TEST_SUITE(Float)
TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(ConfigureOnceUseFromDifferentThreads,
                       CpuActivationFloatThreadSafeFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(),
                               FloatActivationDatasetForThreadSafetyTests,
                               make("DataType", DataType::F32)))
{
    // Validate output
    for (int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i], helper::relative_tolerance(_data_type, _function),
                 helper::tolerance_num(_data_type, _function), helper::absolute_tolerance(_data_type, _function));
    }
}
TEST_SUITE_END() // F32

#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(F16)
FIXTURE_DATA_TEST_CASE(ConfigureOnceUseFromDifferentThreads,
                       CpuActivationFloatThreadSafeFixture<half>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(),
                               FloatActivationDatasetForThreadSafetyTests,
                               make("DataType", DataType::F16)))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        for (int i = 0; i < _num_parallel_runs; ++i)
        {
            validate(Accessor(_target[i]), _reference[i], helper::relative_tolerance(_data_type, _function),
                     helper::tolerance_num(_data_type, _function), helper::absolute_tolerance(_data_type, _function));
        }
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
TEST_SUITE_END() // F16
#endif           // ARM_COMPUTE_ENABLE_FP16

TEST_SUITE_END() // Float

TEST_SUITE(Quantized)

// Int8 and UInt8 are very similar, therefore no need to test both from thread-safety perspective
TEST_SUITE(QASYMM8_SIGNED)
FIXTURE_DATA_TEST_CASE(ConfigureOnceUseFromDifferentThreads,
                       CpuActivationQuantizedThreadSafeFixture<int8_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::SmallShapes(),
                               QuantizedActivationDatasetForThreadSafetyTests,
                               make("DataType", DataType::QASYMM8_SIGNED),
                               make("QuantizationInfo", {QuantizationInfo(0.5f, 10.0f)})))
{
    // Validate output
    for (int i = 0; i < _num_parallel_runs; ++i)
    {
        validate(Accessor(_target[i]), _reference[i], helper::tolerance_qasymm8(_function));
    }
}

TEST_SUITE_END() // QASYMM8_SIGNED
TEST_SUITE_END() // Quantized
TEST_SUITE_END() // ThreadSafety
#endif           // #ifndef BARE_METAL

TEST_SUITE_END() // CpuActivation
TEST_SUITE_END() // OPERATORS
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
