/*
 * Copyright (c) 2024 Arm Limited.
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
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLScatter.h"
#include "tests/validation/fixtures/ScatterLayerFixture.h"
#include "tests/CL/CLAccessor.h"
#include "tests/framework/Macros.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr AbsoluteTolerance<float>   tolerance_f32(0.001f);  /**< Tolerance value for comparing reference's output against implementation's output for 32-bit floating-point type */
constexpr AbsoluteTolerance<float>   tolerance_f16(0.01f);   /**< Tolerance value for comparing reference's output against implementation's output for 16-bit floating-point type */
constexpr AbsoluteTolerance<uint8_t> tolerance_qasymm8(1);   /**< Tolerance value for comparing reference's output against implementation's output for 8-bit asymmetric type */
constexpr AbsoluteTolerance<int8_t>  tolerance_qasymm8_s(1); /**< Tolerance value for comparing reference's output against implementation's output for 8-bit signed asymmetric type */
} // namespace

template <typename T>
using CLScatterLayerFixture = ScatterValidationFixture<CLTensor, CLAccessor, CLScatter, T>;

TEST_SUITE(CL)
TEST_SUITE(ScatterLayer)
TEST_SUITE(Float)
TEST_SUITE(FP32)
TEST_SUITE_END() // FP32
TEST_SUITE_END() // Float
TEST_SUITE_END() // ScatterLayer
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
