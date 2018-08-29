/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/runtime//Tensor.h"
#include "arm_compute/runtime/NEON/functions/NEConvolution.h"
#include "tests/NEON/Accessor.h"
#include "tests/benchmark/fixtures/ConvolutionFixture.h"
#include "tests/datasets/BorderModeDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace test
{
namespace benchmark
{
namespace
{
// *INDENT-OFF*
// clang-format off

#define CONVOLUTION_SQUARE_DATA_TEST_CASE(TEST_NAME, MODE, SHAPES, DT, FILTER_SIZE)                    \
    REGISTER_FIXTURE_DATA_TEST_CASE(TEST_NAME, NEConvolutionFixture, framework::DatasetMode::MODE,     \
                                combine(combine(combine(                                               \
                                datasets::SHAPES,                                                      \
                                framework::dataset::make("DataType", DataType::DT)),                   \
                                datasets::BorderModes()),                                              \
                                framework::dataset::make("FilterSize", { FILTER_SIZE })));

#define CONVOLUTION_RECTANGLE_DATA_TEST_CASE(TEST_NAME, MODE, SHAPES, DT)                              \
    REGISTER_FIXTURE_DATA_TEST_CASE(TEST_NAME, NEConvolutionFixture, framework::DatasetMode::MODE,     \
                                combine(combine(combine(combine(                                       \
                                datasets::SHAPES,                                                      \
                                framework::dataset::make("DataType", DataType::DT)),                   \
                                datasets::BorderModes()),                                              \
                                framework::dataset::make("FilterSize", { 3, 5, 7, 9 })),              \
                                framework::dataset::make("FilterSize", { 3, 5, 7, 9 })));

#define CONVOLUTION_SEPARABLE_DATA_TEST_CASE(TEST_NAME, MODE, SHAPES, DT, FILTER_SIZE)                 \
    CONVOLUTION_SQUARE_DATA_TEST_CASE(TEST_NAME, MODE, SHAPES, DT, FILTER_SIZE)

// clang-format on
// *INDENT-ON*

} // namespace

TEST_SUITE(NEON)
TEST_SUITE(CustomConvolution)

TEST_SUITE(Square3x3)

using NEConvolutionFixture = ConvolutionSquareFixture<Tensor, NEConvolution3x3, Accessor>;

TEST_SUITE(U8)
CONVOLUTION_SQUARE_DATA_TEST_CASE(RunSmall, PRECOMMIT, SmallShapes(), U8, 3)
CONVOLUTION_SQUARE_DATA_TEST_CASE(RunLarge, NIGHTLY, LargeShapes(), U8, 3)
TEST_SUITE_END() // U8

TEST_SUITE(S16)
CONVOLUTION_SQUARE_DATA_TEST_CASE(RunSmall, PRECOMMIT, SmallShapes(), S16, 3)
CONVOLUTION_SQUARE_DATA_TEST_CASE(RunLarge, NIGHTLY, LargeShapes(), S16, 3)
TEST_SUITE_END() // S16

TEST_SUITE_END() // Square3x3

TEST_SUITE(Square5x5)

using NEConvolutionFixture = ConvolutionSquareFixture<Tensor, NEConvolution5x5, Accessor>;

TEST_SUITE(U8)
CONVOLUTION_SQUARE_DATA_TEST_CASE(RunSmall, PRECOMMIT, SmallShapes(), U8, 5)
CONVOLUTION_SQUARE_DATA_TEST_CASE(RunLarge, NIGHTLY, LargeShapes(), U8, 5)
TEST_SUITE_END() // U8

TEST_SUITE(S16)
CONVOLUTION_SQUARE_DATA_TEST_CASE(RunSmall, PRECOMMIT, SmallShapes(), S16, 5)
CONVOLUTION_SQUARE_DATA_TEST_CASE(RunLarge, NIGHTLY, LargeShapes(), S16, 5)
TEST_SUITE_END() // S16

TEST_SUITE_END() // Square5x5

TEST_SUITE(Square7x7)

using NEConvolutionFixture = ConvolutionSquareFixture<Tensor, NEConvolution7x7, Accessor>;

TEST_SUITE(U8)
CONVOLUTION_SQUARE_DATA_TEST_CASE(RunSmall, PRECOMMIT, SmallShapes(), U8, 7)
CONVOLUTION_SQUARE_DATA_TEST_CASE(RunLarge, NIGHTLY, LargeShapes(), U8, 7)
TEST_SUITE_END() // U8

TEST_SUITE(S16)
CONVOLUTION_SQUARE_DATA_TEST_CASE(RunSmall, PRECOMMIT, SmallShapes(), S16, 7)
CONVOLUTION_SQUARE_DATA_TEST_CASE(RunLarge, NIGHTLY, LargeShapes(), S16, 7)
TEST_SUITE_END() // S16

TEST_SUITE_END() // Square7x7

TEST_SUITE(Square9x9)

using NEConvolutionFixture = ConvolutionSquareFixture<Tensor, NEConvolution9x9, Accessor>;

TEST_SUITE(U8)
CONVOLUTION_SQUARE_DATA_TEST_CASE(RunSmall, PRECOMMIT, SmallShapes(), U8, 9)
CONVOLUTION_SQUARE_DATA_TEST_CASE(RunLarge, NIGHTLY, LargeShapes(), U8, 9)
TEST_SUITE_END() // U8

TEST_SUITE(S16)
CONVOLUTION_SQUARE_DATA_TEST_CASE(RunSmall, PRECOMMIT, SmallShapes(), S16, 9)
CONVOLUTION_SQUARE_DATA_TEST_CASE(RunLarge, NIGHTLY, LargeShapes(), S16, 9)
TEST_SUITE_END() // S16

TEST_SUITE_END() // Square9x9

TEST_SUITE(Rectangle)

using NEConvolutionFixture = ConvolutionRectangleFixture<Tensor, NEConvolutionRectangle, Accessor>;

TEST_SUITE(U8)
CONVOLUTION_RECTANGLE_DATA_TEST_CASE(RunSmall, PRECOMMIT, SmallShapes(), U8)
CONVOLUTION_RECTANGLE_DATA_TEST_CASE(RunLarge, NIGHTLY, LargeShapes(), U8)
TEST_SUITE_END() // U8

TEST_SUITE(S16)
CONVOLUTION_RECTANGLE_DATA_TEST_CASE(RunSmall, PRECOMMIT, SmallShapes(), S16)
CONVOLUTION_RECTANGLE_DATA_TEST_CASE(RunLarge, NIGHTLY, LargeShapes(), S16)
TEST_SUITE_END() // S16

TEST_SUITE_END() // Rectangle

TEST_SUITE(Separable5x5)

using NEConvolutionFixture = ConvolutionSeperableFixture<Tensor, NEConvolution5x5, Accessor>;

TEST_SUITE(U8)
CONVOLUTION_SEPARABLE_DATA_TEST_CASE(RunSmall, PRECOMMIT, SmallShapes(), U8, 5)
CONVOLUTION_SEPARABLE_DATA_TEST_CASE(RunLarge, NIGHTLY, LargeShapes(), U8, 5)
TEST_SUITE_END() // U8

TEST_SUITE(S16)
CONVOLUTION_SEPARABLE_DATA_TEST_CASE(RunSmall, PRECOMMIT, SmallShapes(), S16, 5)
CONVOLUTION_SEPARABLE_DATA_TEST_CASE(RunLarge, NIGHTLY, LargeShapes(), S16, 5)
TEST_SUITE_END() // S16

TEST_SUITE_END() // Separable5x5

TEST_SUITE(Separable7x7)

using NEConvolutionFixture = ConvolutionSeperableFixture<Tensor, NEConvolution7x7, Accessor>;

TEST_SUITE(U8)
CONVOLUTION_SEPARABLE_DATA_TEST_CASE(RunSmall, PRECOMMIT, SmallShapes(), U8, 7)
CONVOLUTION_SEPARABLE_DATA_TEST_CASE(RunLarge, NIGHTLY, LargeShapes(), U8, 7)
TEST_SUITE_END() // U8

TEST_SUITE(S16)
CONVOLUTION_SEPARABLE_DATA_TEST_CASE(RunSmall, PRECOMMIT, SmallShapes(), S16, 7)
CONVOLUTION_SEPARABLE_DATA_TEST_CASE(RunLarge, NIGHTLY, LargeShapes(), S16, 7)
TEST_SUITE_END() // S16

TEST_SUITE_END() // Separable7x7

TEST_SUITE(Separable9x9)

using NEConvolutionFixture = ConvolutionSeperableFixture<Tensor, NEConvolution9x9, Accessor>;

TEST_SUITE(U8)
CONVOLUTION_SEPARABLE_DATA_TEST_CASE(RunSmall, PRECOMMIT, SmallShapes(), U8, 9)
CONVOLUTION_SEPARABLE_DATA_TEST_CASE(RunLarge, NIGHTLY, LargeShapes(), U8, 9)
TEST_SUITE_END() // U8

TEST_SUITE(S16)
CONVOLUTION_SEPARABLE_DATA_TEST_CASE(RunSmall, PRECOMMIT, SmallShapes(), S16, 9)
CONVOLUTION_SEPARABLE_DATA_TEST_CASE(RunLarge, NIGHTLY, LargeShapes(), S16, 9)
TEST_SUITE_END() // S16

TEST_SUITE_END() // Separable9x9

TEST_SUITE_END() // CustomConvolution
TEST_SUITE_END() // NEON
} // namespace benchmark
} // namespace test
} // namespace arm_compute
