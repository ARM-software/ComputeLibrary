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
#include "arm_compute/runtime/NEON/functions/NEPadLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/framework/Macros.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/validation/fixtures/PadLayerFixture.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
using framework::dataset::make;

namespace
{
const auto PaddingSizesDataset = make("PaddingSize",
                                      {
                                          PaddingList{{0, 0}},
                                          PaddingList{{1, 1}},
                                          PaddingList{{1, 1}, {2, 2}},
                                          PaddingList{{1, 1}, {1, 1}, {1, 1}},
                                          PaddingList{{0, 0}, {1, 0}, {0, 1}},
                                          PaddingList{{0, 1}, {1, 0}, {0, 1}},
                                      });
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(PadLayer)

// *INDENT-OFF*
// clang-format off

DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(
               make("InputInfo", { TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Mismatching data type input/output
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Mismatching shapes
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Mismatching data type input/output
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),     // Mismatching shapes
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U), 1, DataType::F32)
                                                     }),
               make("OutputInfo",{ TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(28U, 11U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(29U, 17U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(29U, 15U, 4U, 3U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 14U, 3U, 4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U, 3U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 13U, 2U), 1, DataType::F16),
                                                       TensorInfo(TensorShape(28U, 11U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(29U, 17U, 2U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(29U, 15U, 4U, 3U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(27U, 14U, 3U, 4U), 1, DataType::F32),
                                                       TensorInfo(TensorShape(32U, 13U, 2U, 3U), 1, DataType::F32)
                                                     }),
               make("PaddingSize", { PaddingList{{0, 0}},
                                                         PaddingList{{1, 1}},
                                                         PaddingList{{1, 1}, {2, 2}},
                                                         PaddingList{{1,1}, {1,1}, {1,1}, {1,1}},
                                                         PaddingList{{0,0}, {1,0}, {0,1}, {1,2}},
                                                         PaddingList{{0,0}, {0,0}, {0,0}, {1,1}},
                                                         PaddingList{{0, 0}},
                                                         PaddingList{{1, 1}},
                                                         PaddingList{{1, 1}, {2, 2}},
                                                         PaddingList{{1,1}, {1,1}, {1,1}, {1,1}},
                                                         PaddingList{{0,0}, {1,0}, {0,1}, {1,2}},
                                                         PaddingList{{0,0}, {0,0}, {0,0}, {1,1}}
                                                         }),
               make("PaddingMode", { PaddingMode::CONSTANT,
                                                         PaddingMode::CONSTANT,
                                                         PaddingMode::CONSTANT,
                                                         PaddingMode::CONSTANT,
                                                         PaddingMode::CONSTANT,
                                                         PaddingMode::CONSTANT,
                                                         PaddingMode::REFLECT,
                                                         PaddingMode::REFLECT,
                                                         PaddingMode::REFLECT,
                                                         PaddingMode::REFLECT,
                                                         PaddingMode::REFLECT,
                                                         PaddingMode::SYMMETRIC }),
               make("Expected", { false, false, true, true, true, true, false, false, true, false, false, true })
               ),
               input_info, output_info, padding, mode, expected)
{
    ARM_COMPUTE_EXPECT(bool(NEPadLayer::validate(&input_info.clone()->set_is_resizable(true), &output_info.clone()->set_is_resizable(true), padding, PixelValue(), mode)) == expected, framework::LogLevel::ERRORS);
}

// clang-format on
// *INDENT-ON*

template <typename T>
using NEPaddingFixture = PaddingFixture<Tensor, Accessor, NEPadLayer, T>;

TEST_SUITE(Float)

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEPaddingFixture<float>,
                       framework::DatasetMode::ALL,
                       combine(datasets::Small3DShapes(),
                               make("DataType", {DataType::F32}),
                               PaddingSizesDataset,
                               make("PaddingMode", {PaddingMode::CONSTANT, PaddingMode::REFLECT})))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEPaddingFixture<float>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::Large3DShapes(),
                               make("DataType", {DataType::F32}),
                               PaddingSizesDataset,
                               make("PaddingMode",
                                    {PaddingMode::CONSTANT, PaddingMode::REFLECT, PaddingMode::SYMMETRIC})))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // FP32

#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEPaddingFixture<half>,
                       framework::DatasetMode::ALL,
                       combine(datasets::Small3DShapes(),
                               make("DataType", {DataType::F16}),
                               PaddingSizesDataset,
                               make("PaddingMode", {PaddingMode::CONSTANT, PaddingMode::REFLECT})))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEPaddingFixture<half>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::Large3DShapes(),
                               make("DataType", {DataType::F16}),
                               PaddingSizesDataset,
                               make("PaddingMode",
                                    {PaddingMode::CONSTANT, PaddingMode::REFLECT, PaddingMode::SYMMETRIC})))
{
    if (CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference);
    }
    else
    {
        ARM_COMPUTE_TEST_WARNING("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_WARNING();
    }
}
TEST_SUITE_END() // FP16
#endif           /* ARM_COMPUTE_ENABLE_FP16 */
TEST_SUITE_END() // Float

TEST_SUITE(Quantized)
TEST_SUITE(QASYMM8)
FIXTURE_DATA_TEST_CASE(RunSmall,
                       NEPaddingFixture<uint8_t>,
                       framework::DatasetMode::ALL,
                       combine(datasets::Small3DShapes(),
                               make("DataType", {DataType::QASYMM8}),
                               PaddingSizesDataset,
                               make("PaddingMode", {PaddingMode::CONSTANT, PaddingMode::REFLECT})))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
FIXTURE_DATA_TEST_CASE(RunLarge,
                       NEPaddingFixture<uint8_t>,
                       framework::DatasetMode::NIGHTLY,
                       combine(datasets::Large3DShapes(),
                               make("DataType", {DataType::QASYMM8}),
                               PaddingSizesDataset,
                               make("PaddingMode",
                                    {PaddingMode::CONSTANT, PaddingMode::REFLECT, PaddingMode::SYMMETRIC})))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // QASYMM8
TEST_SUITE_END() // Quantized

TEST_SUITE_END() // PadLayer
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
