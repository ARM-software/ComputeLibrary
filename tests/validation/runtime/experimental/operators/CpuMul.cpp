/*
 * Copyright (c) 2017-2021, 2024 Arm Limited.
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
#include "arm_compute/runtime/experimental/operators/CpuMul.h"

#include "arm_compute/core/Rounding.h"

#include "tests/datasets/ConvertPolicyDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/validation/fixtures/CpuMulFixture.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
const float scale_255   = 1.f / 255.f;
const float scale_other = 1.f / 32768.f;

/** Tests for in-place computation
 * With current interface storing TensorInfo with quantization information
 * in the kernel, it is difficult to have different tensor metadata
 * (e.g., quantization information, data type, different shape for broadcasting)
 * when an input is used as the output of the computation.
 * So, the following dataset for in-place computation is used only when
 * the exact same input and output Tensor object makes sense
 * (i.e., all the tensor metadata is the same) whereas if output is
 * expected to have either different quantization information, data type
 * or different shape we are not testing in-place computation.
 */
const auto OutOfPlaceDataSet = framework::dataset::make("InPlace", {false});
} // namespace

using CpuMulU8U8toS16Fixture =
    CpuMulValidationFixture<Tensor, Accessor, experimental::op::CpuMul, uint8_t, uint8_t, int16_t>;

TEST_SUITE(NEON)
TEST_SUITE(CpuMul)

TEST_SUITE(U8U8toS16)
FIXTURE_DATA_TEST_CASE(
    SmokeTest0,
    CpuMulU8U8toS16Fixture,
    framework::DatasetMode::PRECOMMIT,
    combine(combine(combine(combine(combine(combine(combine(datasets::SmallShapes(),
                                                            framework::dataset::make("DataTypeIn1", DataType::U8)),
                                                    framework::dataset::make("DataTypeIn2", DataType::U8)),
                                            framework::dataset::make("DataTypeOut", DataType::S16)),
                                    framework::dataset::make("Scale", {scale_255})),
                            datasets::ConvertPolicies()),
                    framework::dataset::make("RoundingPolicy", RoundingPolicy::TO_NEAREST_UP)),
            OutOfPlaceDataSet))
{
    // Validate output
    validate_wrap(Accessor(_target), _reference, AbsoluteTolerance<int16_t>(1), 0.f);
}

FIXTURE_DATA_TEST_CASE(
    SmokeTest1,
    CpuMulU8U8toS16Fixture,
    framework::DatasetMode::PRECOMMIT,
    combine(combine(combine(combine(combine(combine(combine(datasets::SmallShapes(),
                                                            framework::dataset::make("DataTypeIn1", DataType::U8)),
                                                    framework::dataset::make("DataTypeIn2", DataType::U8)),
                                            framework::dataset::make("DataTypeOut", DataType::S16)),
                                    framework::dataset::make("Scale", {scale_other})),
                            datasets::ConvertPolicies()),
                    framework::dataset::make("RoundingPolicy", RoundingPolicy::TO_ZERO)),
            OutOfPlaceDataSet))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // U8U8toS16

TEST_SUITE_END() // CpuMul
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
