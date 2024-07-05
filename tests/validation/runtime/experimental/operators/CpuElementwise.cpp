/*
 * Copyright (c) 2018-2021, 2024 Arm Limited.
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
#include "arm_compute/runtime/experimental/operators/CpuElementwise.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/validation/fixtures/CpuElementwiseFixture.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
RelativeTolerance<float> tolerance_div_fp32(0.000001f);

const auto ElementwiseFP32Dataset = combine(
    combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::F32)),
    framework::dataset::make("DataType", DataType::F32));

const auto InPlaceDataSet    = framework::dataset::make("InPlace", {false, true});
const auto OutOfPlaceDataSet = framework::dataset::make("InPlace", {false});
} // namespace

TEST_SUITE(NEON)

TEST_SUITE(CpuElementwiseDivision)
template <typename T>
using CpuElementwiseDivisionFixture =
    CpuElementwiseDivisionValidationFixture<Tensor, Accessor, experimental::op::CpuElementwiseDivision, T>;

TEST_SUITE(Float)
TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(SmokeTest,
                       CpuElementwiseDivisionFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(combine(datasets::SmallShapes(), ElementwiseFP32Dataset), InPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference, tolerance_div_fp32, 0.01);
}
TEST_SUITE_END() // F32
TEST_SUITE_END() // Float
TEST_SUITE_END() // CpuElementwiseMin

TEST_SUITE(CpuElementwiseMax)
template <typename T>
using CpuElementwiseMaxFixture =
    CpuElementwiseMaxValidationFixture<Tensor, Accessor, experimental::op::CpuElementwiseMax, T>;

TEST_SUITE(Float)
TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(SmokeTest,
                       CpuElementwiseMaxFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(combine(datasets::SmallShapes(), ElementwiseFP32Dataset), InPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // F32
TEST_SUITE_END() // Float
TEST_SUITE_END() // CpuElementwiseMin

TEST_SUITE(CpuElementwiseMin)

template <typename T>
using CpuElementwiseMinFixture =
    CpuElementwiseMinValidationFixture<Tensor, Accessor, experimental::op::CpuElementwiseMin, T>;

TEST_SUITE(Float)
TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(SmokeTest,
                       CpuElementwiseMinFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(combine(datasets::SmallShapes(), ElementwiseFP32Dataset), InPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // F32
TEST_SUITE_END() // Float
TEST_SUITE_END() // CpuElementwiseMin

TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
