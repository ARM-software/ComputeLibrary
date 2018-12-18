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
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLRange.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ConvertPolicyDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/RangeFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr RelativeTolerance<float> tolerance(0.01f);
constexpr AbsoluteTolerance<float> abs_tolerance(0.02f);
const auto                         start_dataset          = framework::dataset::make("Start", { float(3), float(-17), float(16) });
const auto                         unsigned_start_dataset = framework::dataset::make("Start", { float(3), float(16) });
const auto                         float_step_dataset     = framework::dataset::make("Step", { float(1), float(-0.2f), float(0.2), float(12.2), float(-12.2), float(-1.2), float(-3), float(3) });
const auto                         step_dataset           = framework::dataset::make("Step", { float(1), float(12), float(-12), float(-1), float(-3), float(3) });
const auto                         unsigned_step_dataset  = framework::dataset::make("Step", { float(1), float(12), float(3) });
} // namespace

TEST_SUITE(CL)
TEST_SUITE(Range)

// *INDENT-OFF*
// clang-format off

DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
               framework::dataset::make("OutputInfo", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                        TensorInfo(TensorShape(32U), 1, DataType::U8),
                                                        TensorInfo(TensorShape(27U), 1, DataType::U8),
                                                        TensorInfo(TensorShape(32U), 1, DataType::U8),
                                                        TensorInfo(TensorShape(32U), 1, DataType::F32),
                                                        TensorInfo(TensorShape(27U), 1, DataType::U8),
                                                        TensorInfo(TensorShape(27U), 1, DataType::U8),
                                                        TensorInfo(TensorShape(10U), 1, DataType::U8),
                                                      }),
               framework::dataset::make("Start",{ 0.0f, 15.0f, 1500.0f, 100.0f, -15.0f, 0.2f , 2.0f , 10.0f})),
               framework::dataset::make("End",{ 100.0f, 15.0f, 2500.0f, -1000.0f, 15.0f, 10.0f, 10.0f,100.0f })),
               framework::dataset::make("Step",{ 100.0f, 15.0f, 10.0f, 100.0f, -15.0f, 1.0f, 0.0f, 10.0f })),
               framework::dataset::make("Expected", { false, //1-D tensor expected
                                                    false, //start == end
                                                    false, //output vector size insufficient
                                                    false, //sign of step incorrect
                                                    false, //sign of step incorrect
                                                    false, //data type incompatible
                                                    false, //step = 0
                                                    true,
                                                    })),
               output_info, start, end, step, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLRange::validate(&output_info, start, end, step)) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using CLRangeFixture = RangeFixture<CLTensor, CLAccessor, CLRange, T>;

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLRangeFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(
                                                                                                                 framework::dataset::make("DataType", DataType::U8),
                                                                                                                 unsigned_start_dataset),
                                                                                                             unsigned_step_dataset),
                                                                                                     framework::dataset::make("QuantizationInfo", { QuantizationInfo() })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance, 0.f, abs_tolerance);
}
TEST_SUITE_END() //U8

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)

FIXTURE_DATA_TEST_CASE(RunSmall, CLRangeFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(
                                                                                                                 framework::dataset::make("DataType", DataType::QASYMM8),
                                                                                                                 start_dataset),
                                                                                                             step_dataset),
                                                                                                     framework::dataset::make("QuantizationInfo", { QuantizationInfo(0.3457f, 120.0f) })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance, 0.f, abs_tolerance);
}
TEST_SUITE_END() //QASYMM8
TEST_SUITE_END() //Quantized

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLRangeFixture<int16_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(
                                                                                                                 framework::dataset::make("DataType", DataType::S16),
                                                                                                                 start_dataset),
                                                                                                             step_dataset),
                                                                                                     framework::dataset::make("QuantizationInfo", { QuantizationInfo() })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance, 0.f, abs_tolerance);
}
TEST_SUITE_END() //S16

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLRangeFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(
                                                                                                              framework::dataset::make("DataType", DataType::F16),
                                                                                                              start_dataset),
                                                                                                          float_step_dataset),
                                                                                                  framework::dataset::make("QuantizationInfo", { QuantizationInfo() })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance, 0.f, abs_tolerance);
}
TEST_SUITE_END() //FP16

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, CLRangeFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(
                                                                                                               framework::dataset::make("DataType", DataType::F32),
                                                                                                               start_dataset),
                                                                                                           float_step_dataset),
                                                                                                   framework::dataset::make("QuantizationInfo", { QuantizationInfo() })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, tolerance, 0.f, abs_tolerance);
}
TEST_SUITE_END() //FP32
TEST_SUITE_END() //Float

TEST_SUITE_END() //Range
TEST_SUITE_END() //CL
} // namespace validation
} // namespace test
} // namespace arm_compute
