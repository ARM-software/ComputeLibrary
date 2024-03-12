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
#include "arm_compute/runtime/NEON/functions/NETile.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/TileFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
const auto MultiplesDataset = framework::dataset::make("Multiples", { Multiples{ 3 },
                                                                      Multiples{ 2, 2 },
                                                                      Multiples{ 1, 1, 3, 4 },
                                                                      Multiples{ 2, 1, 2, 2 },
                                                                      Multiples{ 2, 1, 3 },
                                                                      Multiples{ 2, 2, 2 }
                                                                    });
} // namespace
TEST_SUITE(NEON)
TEST_SUITE(Tile)

// *INDENT-OFF*
// clang-format off
DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
        framework::dataset::make("InputInfo", { TensorInfo(TensorShape(10, 10), 1, DataType::F32),
                                                TensorInfo(TensorShape(10, 10), 1, DataType::F32),  // Mismatching shape
                                                TensorInfo(TensorShape(10, 10), 1, DataType::F16), // Mismatching type
                                                TensorInfo(TensorShape(10, 10), 1, DataType::F32)}), // Wrong multiples
        framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(10, 20), 1, DataType::F32),
                                                TensorInfo(TensorShape(20, 20), 1, DataType::F32),
                                                TensorInfo(TensorShape(20, 20), 1, DataType::F32),
                                                TensorInfo(TensorShape(10, 20), 1, DataType::F32)})),
        framework::dataset::make("Multiples",{ Multiples{1, 2}, Multiples{1, 2}, Multiples{0, 1} })),
        framework::dataset::make("Expected", {true, false, false, false })),
        input_info, output_info, multiples, expected)
{
    const Status status = NETile::validate(&input_info.clone()->set_is_resizable(false), &output_info.clone()->set_is_resizable(false), multiples);
    ARM_COMPUTE_EXPECT(bool(status) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*

template <typename T>
using NETileFixture = TileValidationFixture<Tensor, Accessor, NETile, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall, NETileFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType", DataType::F16)),
                                                                                                 MultiplesDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NETileFixture<half>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType", DataType::F16)), MultiplesDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // FP16

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall, NETileFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallShapes(), framework::dataset::make("DataType", DataType::F32)),
                                                                                                  MultiplesDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge, NETileFixture<float>, framework::DatasetMode::NIGHTLY, combine(combine(datasets::LargeShapes(), framework::dataset::make("DataType", DataType::F32)),
                                                                                                MultiplesDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float

TEST_SUITE(Integer)
TEST_SUITE(S8)
FIXTURE_DATA_TEST_CASE(RunSmall, NETileFixture<int8_t>, framework::DatasetMode::ALL,
                       combine(
                           combine(datasets::SmallShapes(), framework::dataset::make("DataType", { DataType::S8 })),
                           MultiplesDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // S8
TEST_SUITE_END() // Integer

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall, NETileFixture<uint8_t>, framework::DatasetMode::ALL,
                       combine(
                           combine(datasets::SmallShapes(), framework::dataset::make("DataType", { DataType::QASYMM8 })),
                           MultiplesDataset))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // Tile
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
