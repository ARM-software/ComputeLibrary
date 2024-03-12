/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEUnstack.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "tests/NEON/Accessor.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/UnstackFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
const auto unstack_axis_dataset  = framework::dataset::make("Axis", -3, 3);
const auto unstack_num_dataset   = framework::dataset::make("Num", 1, 3); // The length of the dimension axis
const auto unstack_dataset_small = datasets::Small3DShapes() * unstack_axis_dataset * unstack_num_dataset;
} //namespace

TEST_SUITE(NEON)
TEST_SUITE(Unstack)

DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
                                                                      framework::dataset::make("InputInfo",
{
    TensorInfo(TensorShape(1U, 9U, 8U), 1, DataType::U8),   // Passes, 1 slice on x axis
    TensorInfo(TensorShape(1U, 2U, 3U), 1, DataType::U8),   // fails because axis > input's rank
    TensorInfo(TensorShape(1U, 2U, 3U), 1, DataType::S32),  // fails axis <  (- input's rank)
    TensorInfo(TensorShape(3U, 7U, 5U), 1, DataType::S32),  // passes, 3 slices along X
    TensorInfo(TensorShape(13U, 7U, 5U), 1, DataType::S16), // fails, too few output slices
    TensorInfo(TensorShape(1U, 2U, 3U), 1, DataType::U8),   // fails mismatching data types
}),
framework::dataset::make("OutputInfo",
{
    std::vector<TensorInfo>{ TensorInfo(TensorShape(9U, 8U), 1, DataType::U8) }, std::vector<TensorInfo>{ TensorInfo(TensorShape(2U, 3U), 1, DataType::U8) }, std::vector<TensorInfo>{ TensorInfo(TensorShape(2U, 3U), 1, DataType::S32) },

    std::vector<TensorInfo>{ TensorInfo(TensorShape(7U, 5U), 1, DataType::S32), TensorInfo(TensorShape(7U, 5U), 1, DataType::S32), TensorInfo(TensorShape(7U, 5U), 1, DataType::S32) }, std::vector<TensorInfo>{ TensorInfo(TensorShape(7U, 5U), 1, DataType::S16) }, std::vector<TensorInfo>{ TensorInfo(TensorShape(9U, 8U), 1, DataType::S32) },
})),
framework::dataset::make("Axis", { -3, 3, -4, -3, 1, 1 })),
framework::dataset::make("Num", { 1, 1, 1, 1, 0, 1 })),
framework::dataset::make("Expected", { true, false, false, true, false, false })),
input_info, output_info, axis, num, expected)
{
    std::vector<TensorInfo>    ti(output_info);
    std::vector<ITensorInfo *> vec(num);
    for(size_t j = 0; j < vec.size(); ++j)
    {
        vec[j] = &ti[j];
    }
    ARM_COMPUTE_EXPECT(bool(NEUnstack::validate(&input_info.clone()->set_is_resizable(false), vec, axis)) == expected, framework::LogLevel::ERRORS);
}

template <typename T>
using NEUnstackFixture = UnstackValidationFixture<Tensor, ITensor, Accessor, NEUnstack, T>;

TEST_SUITE(F32)
FIXTURE_DATA_TEST_CASE(RunSmall, NEUnstackFixture<float>, framework::DatasetMode::PRECOMMIT, unstack_dataset_small * framework::dataset::make("DataType", { DataType::F32 }))
{
    ARM_COMPUTE_ERROR_ON(_target.size() != _reference.size());
    // Validate output
    for(size_t k = 0; k < _target.size(); ++k)
    {
        validate(Accessor(_target[k]), _reference[k]);
    }
}
TEST_SUITE_END() // F32

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(F16)
FIXTURE_DATA_TEST_CASE(RunSmall, NEUnstackFixture<half>, framework::DatasetMode::PRECOMMIT, unstack_dataset_small * framework::dataset::make("DataType", { DataType::F16 }))
{
    ARM_COMPUTE_ERROR_ON(_target.size() != _reference.size());
    // Validate output
    for(size_t k = 0; k < _target.size(); ++k)
    {
        validate(Accessor(_target[k]), _reference[k]);
    }
}
TEST_SUITE_END() // F16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(Quantized)
FIXTURE_DATA_TEST_CASE(RunSmall, NEUnstackFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, unstack_dataset_small * framework::dataset::make("DataType", { DataType::QASYMM8 }))
{
    ARM_COMPUTE_ERROR_ON(_target.size() != _reference.size());
    // Validate output
    for(size_t k = 0; k < _target.size(); ++k)
    {
        validate(Accessor(_target[k]), _reference[k]);
    }
}
TEST_SUITE_END() // QASYMM8

TEST_SUITE_END() // Unstack
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
