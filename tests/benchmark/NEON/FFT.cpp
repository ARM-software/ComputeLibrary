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
#include "arm_compute/runtime/NEON/functions/NEFFT1D.h"
#include "arm_compute/runtime/NEON/functions/NEFFT2D.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/NEON/Accessor.h"
#include "tests/benchmark/fixtures/FFTFixture.h"
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
const auto data_types = framework::dataset::make("DataType", { DataType::F32 });
const auto shapes     = framework::dataset::make("Shapes", { TensorShape(192U, 128U, 64U), TensorShape(224U, 224U) });
} // namespace

using NEFFT1DFixture = FFTFixture<Tensor, NEFFT1D, FFT1DInfo, Accessor>;
using NEFFT2DFixture = FFTFixture<Tensor, NEFFT2D, FFT2DInfo, Accessor>;

TEST_SUITE(CL)

REGISTER_FIXTURE_DATA_TEST_CASE(FFT1D, NEFFT1DFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(shapes, data_types));

REGISTER_FIXTURE_DATA_TEST_CASE(FFT2D, NEFFT2DFixture, framework::DatasetMode::ALL,
                                framework::dataset::combine(shapes, data_types));

TEST_SUITE_END() // CL
} // namespace benchmark
} // namespace test
} // namespace arm_compute
