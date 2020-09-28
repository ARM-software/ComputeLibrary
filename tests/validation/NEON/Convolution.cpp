/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEConvolution.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "tests/NEON/Accessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/BorderModeDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/ConvolutionFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Tolerance value for comparing reference's output against implementation
 *
 * This is due to the fact that NEON target performs multiplication with reciprocal of scale,
 * while reference performs direct division with scale.
 */
constexpr AbsoluteTolerance<uint8_t> tolerance_u8(1);
constexpr AbsoluteTolerance<int16_t> tolerance_s16(1);
} // namespace

TEST_SUITE(NEON)
TEST_SUITE(CustomConvolution)
TEST_SUITE(Square3x3)
template <typename T>
using NEConvolutionFixture = ConvolutionSquareValidationFixture<Tensor, Accessor, NEConvolution3x3, T>;

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunLarge, NEConvolutionFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()),
                                                                                                                 framework::dataset::make("DataType",
                                                                                                                         DataType::U8)),
                                                                                                                 datasets::BorderModes()),
                                                                                                         framework::dataset::make("filter_size", { 3 })))
{
    // Validate output
    validate(Accessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)), tolerance_u8);
}
TEST_SUITE_END() // U8

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunLarge, NEConvolutionFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()),
                                                                                                                 framework::dataset::make("DataType",
                                                                                                                         DataType::S16)),
                                                                                                                 datasets::BorderModes()),
                                                                                                         framework::dataset::make("filter_size", { 3 })))
{
    // Validate output
    validate(Accessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)), tolerance_s16);
}
TEST_SUITE_END() // S16
TEST_SUITE_END() // Square3x3

TEST_SUITE(Square5x5)
template <typename T>
using NEConvolutionFixture = ConvolutionSquareValidationFixture<Tensor, Accessor, NEConvolution5x5, T>;

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunLarge, NEConvolutionFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()),
                                                                                                                 framework::dataset::make("DataType",
                                                                                                                         DataType::U8)),
                                                                                                                 datasets::BorderModes()),
                                                                                                         framework::dataset::make("filter_size", { 5 })))
{
    // Validate output
    validate(Accessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)), tolerance_u8);
}
TEST_SUITE_END() // U8

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunLarge, NEConvolutionFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()),
                                                                                                                 framework::dataset::make("DataType",
                                                                                                                         DataType::S16)),
                                                                                                                 datasets::BorderModes()),
                                                                                                         framework::dataset::make("filter_size", { 5 })))
{
    // Validate output
    validate(Accessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)), tolerance_s16);
}
TEST_SUITE_END() // S16
TEST_SUITE_END() // Square5x5

TEST_SUITE(Square7x7)
template <typename T>
using NEConvolutionFixture = ConvolutionSquareValidationFixture<Tensor, Accessor, NEConvolution7x7, T>;

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunLarge, NEConvolutionFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()),
                                                                                                                 framework::dataset::make("DataType",
                                                                                                                         DataType::U8)),
                                                                                                                 datasets::BorderModes()),
                                                                                                         framework::dataset::make("filter_size", { 7 })))
{
    // Validate output
    validate(Accessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)), tolerance_u8);
}
TEST_SUITE_END() // U8

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunLarge, NEConvolutionFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()),
                                                                                                                 framework::dataset::make("DataType",
                                                                                                                         DataType::S16)),
                                                                                                                 datasets::BorderModes()),
                                                                                                         framework::dataset::make("filter_size", { 7 })))
{
    // Validate output
    validate(Accessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)), tolerance_s16);
}
TEST_SUITE_END() // S16
TEST_SUITE_END() // Square7x7

TEST_SUITE(Square9x9)
template <typename T>
using NEConvolutionFixture = ConvolutionSquareValidationFixture<Tensor, Accessor, NEConvolution9x9, T>;

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunLarge, NEConvolutionFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()),
                                                                                                                 framework::dataset::make("DataType",
                                                                                                                         DataType::U8)),
                                                                                                                 datasets::BorderModes()),
                                                                                                         framework::dataset::make("filter_size", { 9 })))
{
    // Validate output
    validate(Accessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)), tolerance_u8);
}
TEST_SUITE_END() // U8

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunLarge, NEConvolutionFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()),
                                                                                                                 framework::dataset::make("DataType",
                                                                                                                         DataType::S16)),
                                                                                                                 datasets::BorderModes()),
                                                                                                         framework::dataset::make("filter_size", { 9 })))
{
    // Validate output
    validate(Accessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)), tolerance_s16);
}
TEST_SUITE_END() // S16
TEST_SUITE_END() // Square9x9

TEST_SUITE(Rectangle)
template <typename T>
using NEConvolutionFixture = ConvolutionRectangleValidationFixture<Tensor, Accessor, NEConvolutionRectangle, T>;

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunLarge, NEConvolutionFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()),
                                                                                                                 framework::dataset::make("DataType",
                                                                                                                         DataType::U8)),
                                                                                                                 datasets::BorderModes()),
                                                                                                                 framework::dataset::make("filter_width", { 3, 5, 7, 9 })),
                                                                                                         framework::dataset::make("filter_height", { 3, 5, 7, 9 })))
{
    // Validate output
    validate(Accessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)), tolerance_u8);
}
TEST_SUITE_END() // U8

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunLarge, NEConvolutionFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()),
                                                                                                                 framework::dataset::make("DataType",
                                                                                                                         DataType::S16)),
                                                                                                                 datasets::BorderModes()),
                                                                                                                 framework::dataset::make("filter_width", { 3, 5, 7, 9 })),
                                                                                                         framework::dataset::make("filter_height", { 3, 5, 7, 9 })))
{
    // Validate output
    validate(Accessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)), tolerance_s16);
}
TEST_SUITE_END() // S16
TEST_SUITE_END() // Rectangle

TEST_SUITE(Separable5x5)
template <typename T>
using NEConvolutionFixture = ConvolutionSeparableValidationFixture<Tensor, Accessor, NEConvolution5x5, T>;

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunLarge, NEConvolutionFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()),
                                                                                                                 framework::dataset::make("DataType",
                                                                                                                         DataType::U8)),
                                                                                                                 datasets::BorderModes()),
                                                                                                         framework::dataset::make("filter_size", { 5 })))
{
    // Validate output
    validate(Accessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)), tolerance_u8);
}
TEST_SUITE_END() // U8

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunLarge, NEConvolutionFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()),
                                                                                                                 framework::dataset::make("DataType",
                                                                                                                         DataType::S16)),
                                                                                                                 datasets::BorderModes()),
                                                                                                         framework::dataset::make("filter_size", { 5 })))
{
    // Validate output
    validate(Accessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)), tolerance_s16);
}
TEST_SUITE_END() // S16
TEST_SUITE_END() // Separable5x5

TEST_SUITE(Separable7x7)
template <typename T>
using NEConvolutionFixture = ConvolutionSeparableValidationFixture<Tensor, Accessor, NEConvolution7x7, T>;

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunLarge, NEConvolutionFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()),
                                                                                                                 framework::dataset::make("DataType",
                                                                                                                         DataType::U8)),
                                                                                                                 datasets::BorderModes()),
                                                                                                         framework::dataset::make("filter_size", { 7 })))
{
    // Validate output
    validate(Accessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)), tolerance_u8);
}
TEST_SUITE_END() // U8

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunLarge, NEConvolutionFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()),
                                                                                                                 framework::dataset::make("DataType",
                                                                                                                         DataType::S16)),
                                                                                                                 datasets::BorderModes()),
                                                                                                         framework::dataset::make("filter_size", { 7 })))
{
    // Validate output
    validate(Accessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)), tolerance_s16);
}
TEST_SUITE_END() // S16
TEST_SUITE_END() // Separable7x7

TEST_SUITE(Separable9x9)
template <typename T>
using NEConvolutionFixture = ConvolutionSeparableValidationFixture<Tensor, Accessor, NEConvolution9x9, T>;

TEST_SUITE(U8)
FIXTURE_DATA_TEST_CASE(RunLarge, NEConvolutionFixture<uint8_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()),
                                                                                                                 framework::dataset::make("DataType",
                                                                                                                         DataType::U8)),
                                                                                                                 datasets::BorderModes()),
                                                                                                         framework::dataset::make("filter_size", { 9 })))
{
    // Validate output
    validate(Accessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)), tolerance_u8);
}
TEST_SUITE_END() // U8

TEST_SUITE(S16)
FIXTURE_DATA_TEST_CASE(RunLarge, NEConvolutionFixture<int16_t>, framework::DatasetMode::NIGHTLY, combine(combine(combine(concat(datasets::SmallShapes(), datasets::LargeShapes()),
                                                                                                                 framework::dataset::make("DataType",
                                                                                                                         DataType::S16)),
                                                                                                                 datasets::BorderModes()),
                                                                                                         framework::dataset::make("filter_size", { 9 })))
{
    // Validate output
    validate(Accessor(_target), _reference, shape_to_valid_region(_reference.shape(), (_border_mode == BorderMode::UNDEFINED), BorderSize(_height / 2, _width / 2)), tolerance_s16);
}
TEST_SUITE_END() // S16
TEST_SUITE_END() // Separable9x9

TEST_SUITE_END() // CustomConvolution
TEST_SUITE_END() // NEON
} // namespace validation
} // namespace test
} // namespace arm_compute
