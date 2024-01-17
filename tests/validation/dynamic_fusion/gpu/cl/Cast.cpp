/*
 * Copyright (c) 2022-2024 Arm Limited.
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
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuCast.h"
#include "arm_compute/runtime/CL/CLTensor.h"

#include "tests/CL/CLAccessor.h"
#include "tests/datasets/ConvertPolicyDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/dynamic_fusion/operators/CastFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
// Tolerance
constexpr AbsoluteTolerance<float> zero_tolerance(0);

/** Input data sets **/

// F16
const auto CastF16toF32Dataset = combine(framework::dataset::make("DataType", DataType::F16), framework::dataset::make("DataType", DataType::F32));

// F32
const auto CastF32toF16Dataset = combine(framework::dataset::make("DataType", DataType::F32), framework::dataset::make("DataType", DataType::F16));

class DFConvertPolicies final : public framework::dataset::ContainerDataset<std::vector<ConvertPolicy>>
{
public:
    DFConvertPolicies()
        : ContainerDataset("ConvertPolicy",
    {
        ConvertPolicy::WRAP
    })
    {
    }
};
} // namespace

TEST_SUITE(CL)
TEST_SUITE(DYNAMIC_FUSION)
TEST_SUITE(CAST)

template <typename T>
using DynamicFusionCLCastToF16Fixture = DynamicFusionCastValidationFixture<CLTensor, CLAccessor, GpuCast, T, half>;
template <typename T>
using DynamicFusionCLCastToF32Fixture = DynamicFusionCastValidationFixture<CLTensor, CLAccessor, GpuCast, T, float>;

#define CAST_SUITE(NAME, idt, odt, type, dataset, tolerance)                                                                     \
    TEST_SUITE(NAME)                                                                                                             \
    FIXTURE_DATA_TEST_CASE(RunSmall, type, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), dataset), \
                                                                                              DFConvertPolicies()))              \
    {                                                                                                                            \
        validate(CLAccessor(_target), _reference, tolerance);                                                                    \
    }                                                                                                                            \
    TEST_SUITE_END()

// F16
CAST_SUITE(F16_to_F32, DataType::F16, DataType::F32, DynamicFusionCLCastToF32Fixture<half>, CastF16toF32Dataset, zero_tolerance)

// F32
CAST_SUITE(F32_to_F16, DataType::F32, DataType::F16, DynamicFusionCLCastToF16Fixture<float>, CastF32toF16Dataset, zero_tolerance)

TEST_SUITE_END() // CAST
TEST_SUITE_END() // DYNAMIC_FUSION
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
