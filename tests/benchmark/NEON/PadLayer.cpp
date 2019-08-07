/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEPadLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/benchmark/fixtures/PadLayerFixture.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/datasets/SplitDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"

namespace arm_compute
{
namespace test
{
namespace benchmark
{
namespace
{
const auto Fssd_25_8bit_ShapesDataset = framework::dataset::make("TensorShape",
{
    TensorShape{ 320U, 320U, 3U },
    TensorShape{ 160U, 160U, 16U },
    TensorShape{ 80U, 80U, 32U },
    TensorShape{ 40U, 40U, 64U },
    TensorShape{ 20U, 20U, 128U },
    TensorShape{ 10U, 10U, 256U },
    TensorShape{ 10U, 10U, 64U },
    TensorShape{ 5U, 5U, 32U },
    TensorShape{ 3U, 3U, 32U },
    TensorShape{ 2U, 2U, 32U }
});

const auto PaddingSizesDataset = framework::dataset::make("PaddingSize",
{
    PaddingList{ { 1, 1 }, { 1, 1 } },
});
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(PadLayer)

template <typename T>
using NEPaddingFixture = PaddingFixture<Tensor, Accessor, NEPadLayer, T>;

REGISTER_FIXTURE_DATA_TEST_CASE(RunF32, NEPaddingFixture<float>, framework::DatasetMode::ALL,
                                combine(combine(combine(
                                                    Fssd_25_8bit_ShapesDataset,
                                                    framework::dataset::make("DataType", { DataType::F32 })),
                                                PaddingSizesDataset),
                                        framework::dataset::make("PaddingMode", { PaddingMode::CONSTANT, PaddingMode::REFLECT })));

REGISTER_FIXTURE_DATA_TEST_CASE(RunQASYMM8, NEPaddingFixture<uint8_t>, framework::DatasetMode::ALL,
                                combine(combine(combine(
                                                    Fssd_25_8bit_ShapesDataset,
                                                    framework::dataset::make("DataType", { DataType::QASYMM8 })),
                                                PaddingSizesDataset),
                                        framework::dataset::make("PaddingMode", { PaddingMode::CONSTANT, PaddingMode::REFLECT })));

TEST_SUITE_END() // PadLayer
TEST_SUITE_END() // NEON
} // namespace benchmark
} // namespace test
} // namespace arm_compute
