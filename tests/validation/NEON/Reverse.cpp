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
#include "arm_compute/runtime/NEON/functions/NEReverse.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ReverseFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
auto run_small_dataset = combine(datasets::SmallShapes(), datasets::Tiny1DShapes());
auto run_large_dataset = combine(datasets::LargeShapes(), datasets::Tiny1DShapes());

} // namespace
TEST_SUITE(NEON)
TEST_SUITE(Reverse)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
        framework::dataset::make("InputInfo", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S8), // Invalid axis datatype
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8), // Invalid axis shape
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8), // Invalid axis length (> 4)
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8), // Mismatching shapes
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                            TensorInfo(TensorShape(2U), 1, DataType::U8),
        }),
        framework::dataset::make("OutputInfo", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S8),
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                            TensorInfo(TensorShape(2U, 13U, 2U), 1, DataType::U8),
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                            TensorInfo(TensorShape(2U), 1, DataType::U8),
        })),
        framework::dataset::make("AxisInfo", { TensorInfo(TensorShape(3U), 1, DataType::U8),
                                           TensorInfo(TensorShape(2U, 10U), 1, DataType::U32),
                                           TensorInfo(TensorShape(8U), 1, DataType::U32),
                                           TensorInfo(TensorShape(2U), 1, DataType::U32),
                                           TensorInfo(TensorShape(2U), 1, DataType::U32),
                                           TensorInfo(TensorShape(2U), 1, DataType::U32),
        })),
        framework::dataset::make("Expected", { false, false, false, false, true, true})),
        src_info, dst_info, axis_info, expected)
{
    Status s = NEReverse::validate(&src_info.clone()->set_is_resizable(false),
                                  &dst_info.clone()->set_is_resizable(false),
                                  &axis_info.clone()->set_is_resizable(false));
    ARM_COMPUTE_EXPECT(bool(s) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NEReverseFixture = ReverseValidationFixture<Tensor, Accessor, NEReverse, T>;

TEST_SUITE(Float)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(F16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEReverseFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(run_small_dataset, framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEReverseFixture<half>,
                       framework::DatasetMode::NIGHTLY,
                       combine(run_large_dataset, framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // F16
#endif           /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEReverseFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(run_small_dataset, framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEReverseFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(run_large_dataset, framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // F32
TEST_SUITE_END() // Float

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEReverseFixture<uint8_t>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(run_small_dataset, framework::dataset::make("DataType", DataType::QASYMM8)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEReverseFixture<uint8_t>,
                       framework::DatasetMode::NIGHTLY,
                       combine(run_large_dataset, framework::dataset::make("DataType", DataType::QASYMM8)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // Reverse
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
