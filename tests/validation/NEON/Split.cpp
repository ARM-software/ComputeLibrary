/*
 * Copyright (c) 2018-2021, 2024-2026 Arm Limited.
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

#include "tests/datasets/SplitDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/validation/fixtures/SplitFixture.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
using framework::dataset::make;

TEST_SUITE(NEON)
TEST_SUITE(Split)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(
        make("InputInfo", { TensorInfo(TensorShape(27U, 3U, 16U, 2U), 1, DataType::F32), // Invalid axis
                                                TensorInfo(TensorShape(27U, 3U, 16U, 2U), 1, DataType::F32), // Invalid number of splits
                                                TensorInfo(TensorShape(27U, 3U, 16U, 2U), 1, DataType::F32)
        }),
        make("Axis", { 4, 2, 2 }),
        make("Splits", { 4, 5, 4 }),
        make("Expected", { false, false, true })
                                                          ),
               input_info, axis, splits, expected)
{
    std::vector<TensorInfo> outputs_info(splits);
    std::vector<ITensorInfo*> outputs_info_ptr;
    outputs_info_ptr.reserve(splits);
    for(auto &output_info : outputs_info)
    {
        outputs_info_ptr.emplace_back(&output_info);
    }
    const Status status = NESplit::validate(&input_info.clone()->set_is_resizable(false), outputs_info_ptr, axis);
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}

DATA_TEST_CASE(ValidateSplitShapes, framework::DatasetMode::ALL, zip(
        make("InputInfo", { TensorInfo(TensorShape(27U, 3U, 16U, 2U), 1, DataType::F32),
                                                TensorInfo(TensorShape(27U, 3U, 16U, 2U), 1, DataType::F32)
        }),
        make("Axis", { 2, 2 }),
        make("Splits", { std::vector<TensorInfo>{TensorInfo(TensorShape(27U, 3U, 4U,  2U), 1, DataType::F32),
                                                                     TensorInfo(TensorShape(27U, 3U, 4U,  2U), 1, DataType::F32),
                                                                     TensorInfo(TensorShape(27U, 3U, 8U,  2U), 1, DataType::F32)},
                                             std::vector<TensorInfo>{TensorInfo(TensorShape(27U, 3U, 3U,  2U), 1, DataType::F32),
                                                                     TensorInfo(TensorShape(27U, 3U, 13U, 2U), 1, DataType::F32)} }),
        make("Expected", { true, true })
        ),
        input_info, axis, splits, expected)
{
    std::vector<ITensorInfo*> outputs_info_ptr;

    for(auto &split : splits)
    {
        outputs_info_ptr.emplace_back(const_cast<TensorInfo*>(&split));
    }
    const Status status = NESplit::validate(&input_info.clone()->set_is_resizable(false), outputs_info_ptr, axis);
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NESplitFixture = SplitFixture<Tensor, ITensor, Accessor, NESplit, T>;

template <typename T>
using NESplitShapesFixture = SplitShapesFixture<Tensor, ITensor, Accessor, NESplit, T>;

TEST_SUITE(Float)
#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NESplitFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallSplitDataset(), make("DataType", DataType::F16)))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate outputs
        for (unsigned int i = 0; i < _target.size(); ++i)
        {
            validate(Accessor(_target[i]), _reference[i]);
        }
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NESplitFixture<half>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeSplitDataset(), make("DataType", DataType::F16)))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate outputs
        for (unsigned int i = 0; i < _target.size(); ++i)
        {
            validate(Accessor(_target[i]), _reference[i]);
        }
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
TEST_SUITE_END() // FP16
#endif           /* ARM_COMPUTE_ENABLE_FP16 */

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NESplitFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallSplitDataset(), make("DataType", DataType::F32)))
{
    // Validate outputs
    for (unsigned int i = 0; i < _target.size(); ++i)
    {
        validate(Accessor(_target[i]), _reference[i]);
    }
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NESplitFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::LargeSplitDataset(), make("DataType", DataType::F32)))
{
    // Validate outputs
    for (unsigned int i = 0; i < _target.size(); ++i)
    {
        validate(Accessor(_target[i]), _reference[i]);
    }
}

FIXTURE_DATA_TEST_CASE(RunSmallSplitShapes,
                       NESplitShapesFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(datasets::SmallSplitShapesDataset(), make("DataType", DataType::F32)))
{
    // Validate outputs
    for (unsigned int i = 0; i < _target.size(); ++i)
    {
        validate(Accessor(_target[i]), _reference[i]);
    }
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

TEST_SUITE_END() // Split
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
