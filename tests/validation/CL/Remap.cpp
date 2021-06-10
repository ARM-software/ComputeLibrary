/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLRemap.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/BorderModeDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/RemapFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr AbsoluteTolerance<uint8_t> tolerance_value(1);
} // namespace

TEST_SUITE(CL)
TEST_SUITE(Remap)

// *INDENT-OFF*
// clang-format off

DATA_TEST_CASE(Validate, framework::DatasetMode::ALL, zip(zip(zip(zip(zip(zip(
               framework::dataset::make("input", { TensorInfo(TensorShape(10U, 10U), 1, DataType::U8, DataLayout::NCHW),
                                                   TensorInfo(TensorShape(10U, 10U), 1, DataType::U8, DataLayout::NHWC),
                                                   TensorInfo(TensorShape(10U, 10U), 1, DataType::F16, DataLayout::NCHW),
                                                   TensorInfo(TensorShape(10U, 10U), 1, DataType::F16, DataLayout::NHWC)
                                                      }),
               framework::dataset::make("map_x", { TensorInfo(TensorShape(10U, 10U), 1, DataType::F32, DataLayout::NCHW),
                                                   TensorInfo(TensorShape(10U, 10U), 1, DataType::F32, DataLayout::NHWC),
                                                   TensorInfo(TensorShape(10U, 10U), 1, DataType::F32, DataLayout::NCHW),
                                                   TensorInfo(TensorShape(10U, 10U), 1, DataType::F32, DataLayout::NHWC)

                                                      })),
               framework::dataset::make("map_y", { TensorInfo(TensorShape(10U, 10U), 1, DataType::F32, DataLayout::NCHW),
                                                   TensorInfo(TensorShape(10U, 10U), 1, DataType::F32, DataLayout::NHWC),
                                                   TensorInfo(TensorShape(10U, 10U), 1, DataType::F32, DataLayout::NCHW),
                                                   TensorInfo(TensorShape(10U, 10U), 1, DataType::F32, DataLayout::NHWC)
                                                      })),
               framework::dataset::make("output", { TensorInfo(TensorShape(10U, 10U), 1, DataType::U8, DataLayout::NCHW),
                                                    TensorInfo(TensorShape(10U, 10U), 1, DataType::U8, DataLayout::NHWC),
                                                    TensorInfo(TensorShape(10U, 10U), 1, DataType::F16, DataLayout::NCHW),
                                                    TensorInfo(TensorShape(10U, 10U), 1, DataType::F16, DataLayout::NHWC)
                                                      })),
               framework::dataset::make("policy",{ InterpolationPolicy::NEAREST_NEIGHBOR,
                                                   InterpolationPolicy::NEAREST_NEIGHBOR,
                                                   InterpolationPolicy::NEAREST_NEIGHBOR,
                                                   InterpolationPolicy::NEAREST_NEIGHBOR
                                                      })),
               framework::dataset::make("border_mode",{ BorderMode::CONSTANT,
                                                        BorderMode::CONSTANT,
                                                        BorderMode::CONSTANT,
                                                        BorderMode::CONSTANT
                                                      })),
               framework::dataset::make("Expected", { true, // NCHW, U8
                                                      true, // NHWC, U8
                                                      false, // NCHW, F16
                                                      true // NHWC, F16
                                                    })),
               input, map_x, map_y, output, policy, border_mode, expected)
{
    ARM_COMPUTE_EXPECT(bool(CLRemap::validate(&input, &map_x, &map_y, &output, policy, border_mode, PixelValue{})) == expected, framework::LogLevel::ERRORS);
}
// clang-format on
// *INDENT-ON*
template <typename T>
using CLRemapFixture = RemapValidationFixture<CLTensor, CLAccessor, CLRemap, T>;
template <typename T>
using CLRemapLayoutFixture = RemapValidationMixedLayoutFixture<CLTensor, CLAccessor, CLRemap, T>;

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunSmall, CLRemapLayoutFixture<uint8_t>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                   framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                                   framework::dataset::make("DataType", DataType::U8)),
                                                                                                                   framework::dataset::make("BorderModes", { BorderMode::UNDEFINED, BorderMode::CONSTANT })),
                                                                                                           framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, _valid_mask, tolerance_value);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLRemapFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(datasets::LargeShapes(), framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                           framework::dataset::make("DataType", DataType::U8)),
                                                                                                   framework::dataset::make("BorderModes", { BorderMode::UNDEFINED, BorderMode::CONSTANT })))
{
    // Validate output
    validate(CLAccessor(_target), _reference, _valid_mask, tolerance_value);
}
TEST_SUITE_END() // U8

TEST_SUITE(F16)
FIXTURE_DATA_TEST_CASE(RunSmall, CLRemapLayoutFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(combine(combine(datasets::SmallShapes(),
                                                                                                                        framework::dataset::make("InterpolationPolicy", { InterpolationPolicy::NEAREST_NEIGHBOR, InterpolationPolicy::BILINEAR })),
                                                                                                                        framework::dataset::make("DataType", DataType::F16)),
                                                                                                                framework::dataset::make("BorderModes", { BorderMode::UNDEFINED, BorderMode::CONSTANT })),
                                                                                                        framework::dataset::make("DataLayout", DataLayout::NHWC)))
{
    // Validate output
    validate(CLAccessor(_target), _reference, _valid_mask, tolerance_value);
}
TEST_SUITE_END() // F16
TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
