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
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/functions/CLPadLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/Globals.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/PadLayerFixture.h"
#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
const auto PaddingSizesDataset = framework::dataset::make("PaddingSize", { PaddingList{ { 0, 0 } },
    PaddingList{ { 1, 1 } },
    PaddingList{ { 1, 1 }, { 2, 2 } },
    PaddingList{ { 1, 1 }, { 1, 1 }, { 1, 1 }, { 1, 1 } },
    PaddingList{ { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 2 } },
    PaddingList{ { 0, 0 }, { 0, 0 }, { 0, 0 }, { 1, 1 } }
});
} // namespace

TEST_SUITE(CL)
TEST_SUITE(PadLayer)

// *INDENT-OFF*
// clang-format off

DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(
               framework::dataset::make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Mismatching data type input/output
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Mismatching shapes
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32)
                                                     }),
               framework::dataset::make("OutputInfo",{ TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(28U, 11U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(29U, 17U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(29U, 15U, 4U, 3U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 14U, 3U, 4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U, 3U), 1, DataType::F32)
                                                     })),
               framework::dataset::make("PaddingSize", { PaddingList{{0, 0}},
                                                      PaddingList{{1, 1}},
                                                      PaddingList{{1, 1}, {2, 2}},
                                                      PaddingList{{1,1}, {1,1}, {1,1}, {1,1}},
                                                      PaddingList{{0,0}, {1,0}, {0,1}, {1,2}},
                                                      PaddingList{{0,0}, {0,0}, {0,0}, {1,1}}
                                                      })),
               framework::dataset::make("Expected", { false, false, true, true, true, true })),
               input_info, output_info, padding, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLPadLayer::validate(&input_info.clone()->set_is_resizable(true), &output_info.clone()->set_is_resizable(true), padding)) == expected, framework::LogLevel::ERRORS);
}

// clang-format on
// *INDENT-ON*

template <typename T>
using CLPaddingFixture = PaddingFixture<CLTensor, CLAccessor, CLPadLayer, T>;

TEST_SUITE(Float)

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunPadding, CLPaddingFixture<float>, framework::DatasetMode::ALL,
                       combine(
                           combine(datasets::SmallShapes(), framework::dataset::make("DataType", { DataType::F32 })),
                           PaddingSizesDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // FP32

TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunPadding, CLPaddingFixture<half>, framework::DatasetMode::ALL,
                       combine(
                           combine(datasets::SmallShapes(), framework::dataset::make("DataType", { DataType::F16 })),
                           PaddingSizesDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // FP16
TEST_SUITE_END() // Float

TEST_SUITE(Integer)
TEST_SUITE(S8)
FIXTURE_DATA_TEST_CASE(RunPadding, CLPaddingFixture<int8_t>, framework::DatasetMode::ALL,
                       combine(
                           combine(datasets::SmallShapes(), framework::dataset::make("DataType", { DataType::S8 })),
                           PaddingSizesDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // S8
TEST_SUITE_END() // Integer

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunPadding, CLPaddingFixture<uint8_t>, framework::DatasetMode::ALL,
                       combine(
                           combine(datasets::SmallShapes(), framework::dataset::make("DataType", { DataType::QASYMM8 })),
                           PaddingSizesDataset))
{
    // Validate output
    validate(CLAccessor(_target), _reference);
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // PadLayer
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
