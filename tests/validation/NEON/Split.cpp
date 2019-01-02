/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NESplit.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "tests/NEON/Accessor.h"
#include "tests/datasets/SplitDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/SplitFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(NEON)
TEST_SUITE(Split)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
        framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 3U, 16U, 2U), 1, DataType::F32), // Invalid axis
                                                TensorInfo(TensorShape(27U, 3U, 16U, 2U), 1, DataType::F32), // Invalid number of splits
                                                TensorInfo(TensorShape(27U, 3U, 16U, 2U), 1, DataType::F32)
        }),
        framework::dataset::make("Axis", { 4, 2, 2 })),
                                                              framework::dataset::make("Splits", { 4, 5, 4 })),
                                                          framework::dataset::make("Expected", { false, false, true })),
               input_info, axis, splits, expected)
{
    std::vector<TensorInfo> outputs_info(splits);
    std::vector<ITensorInfo*> outputs_info_ptr;
    for(auto &output_info : outputs_info)
    {
        outputs_info_ptr.emplace_back(&output_info);
    }
    const Status status = NESplit::validate(&input_info.clone()->set_is_resizable(false), outputs_info_ptr, axis);
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

DATA_TEST_CASE(Configuration,
               framework::DatasetMode::ALL,
               combine(datasets::SmallSplitDataset(), framework::dataset::make("DataType", { DataType::QASYMM8, DataType::F32 })),
               shape, axis, splits, data_type)
{
    // Create tensors
    Tensor                 src = create_tensor<Tensor>(shape, data_type);
    std::vector<Tensor>    dsts(splits);
    std::vector<ITensor *> dsts_ptrs;
    for(auto &dst : dsts)
    {
        dsts_ptrs.emplace_back(&dst);
    }

    // Create and Configure function
    NESplit split;
    split.configure(&src, dsts_ptrs, axis);

    // Validate valid regions
    for(auto &dst : dsts)
    {
        const ValidRegion valid_region = shape_to_valid_region(dst.info()->tensor_shape());
        validate(dst.info()->valid_region(), valid_region);
    }
}

template <typename T>
using NESplitFixture = SplitFixture<Tensor, ITensor, Accessor, NESplit, T>;

TEST_SUITE(Float)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NESplitFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallSplitDataset(), framework::dataset::make("DataType", DataType::F16)))
{
    // Validate outputs
    for(unsigned int i = 0; i < _target.size(); ++i)
    {
        validate(Accessor(_target[i]), _reference[i]);
    }
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NESplitFixture<half>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeSplitDataset(), framework::dataset::make("DataType", DataType::F16)))
{
    // Validate outputs
    for(unsigned int i = 0; i < _target.size(); ++i)
    {
        validate(Accessor(_target[i]), _reference[i]);
    }
}
TEST_SUITE_END() // FP16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NESplitFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallSplitDataset(), framework::dataset::make("DataType", DataType::F32)))
{
    // Validate outputs
    for(unsigned int i = 0; i < _target.size(); ++i)
    {
        validate(Accessor(_target[i]), _reference[i]);
    }
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NESplitFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeSplitDataset(), framework::dataset::make("DataType", DataType::F32)))
{
    // Validate outputs
    for(unsigned int i = 0; i < _target.size(); ++i)
    {
        validate(Accessor(_target[i]), _reference[i]);
    }
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

TEST_SUITE_END() // Split
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
