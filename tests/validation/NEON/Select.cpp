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
#include "arm_compute/runtime/NEON/functions/NESelect.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/SelectFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
auto run_small_dataset = combine(datasets::SmallShapes(), framework::dataset::make("has_same_rank", { false, true }));
auto run_large_dataset = combine(datasets::LargeShapes(), framework::dataset::make("has_same_rank", { false, true }));
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(Select)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(
        framework::dataset::make("CInfo", { TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S8), // Invalid condition datatype
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8), // Invalid output datatype
                                            TensorInfo(TensorShape(13U), 1, DataType::U8),          // Invalid c shape
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8), // Mismatching shapes
                                            TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                            TensorInfo(TensorShape(2U), 1, DataType::U8),
        }),
        framework::dataset::make("XInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                           TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                           TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                           TensorInfo(TensorShape(32U, 10U, 2U), 1, DataType::F32),
                                           TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                           TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
        })),
        framework::dataset::make("YInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                           TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                           TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                           TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                           TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                           TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
        })),
        framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::S8),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::U8),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
        })),
        framework::dataset::make("Expected", { false, false, false, false, true, true})),
        c_info, x_info, y_info, output_info, expected)
{
    Status s = NESelect::validate(&c_info.clone()->set_is_resizable(false),
                                  &x_info.clone()->set_is_resizable(false),
                                  &y_info.clone()->set_is_resizable(false),
                                  &output_info.clone()->set_is_resizable(false));
    ARM_COMPUTE_EXPECT(bool(s) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NESelectFixture = SelectValidationFixture<Tensor, Accessor, NESelect, T>;

TEST_SUITE(Float)

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_SUITE(F16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NESelectFixture<half>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(run_small_dataset, framework::dataset::make("DataType", DataType::F16)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NESelectFixture<half>,
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
                       NESelectFixture<float>,
                       framework::DatasetMode::PRECOMMIT,
                       combine(run_small_dataset, framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge,
                       NESelectFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(run_large_dataset, framework::dataset::make("DataType", DataType::F32)))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

TEST_SUITE_END() // Select
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
