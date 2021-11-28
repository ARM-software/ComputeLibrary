/*
 * Copyright (c) 2022 Arm Limited.
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
#include "arm_compute/core/Helpers.h"
#include "src/cpu/kernels/elementwise_binary/generic/neon/impl.h"
namespace arm_compute
{
namespace cpu
{
template <ArithmeticOperation op>
void neon_s32_elementwise_binary(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    return elementwise_arithm_op<op, typename wrapper::traits::neon_vector<int32_t, 4>>(in1, in2, out, window);
}

template void neon_s32_elementwise_binary<ArithmeticOperation::ADD>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_s32_elementwise_binary<ArithmeticOperation::SUB>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_s32_elementwise_binary<ArithmeticOperation::DIV>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_s32_elementwise_binary<ArithmeticOperation::MIN>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_s32_elementwise_binary<ArithmeticOperation::MAX>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_s32_elementwise_binary<ArithmeticOperation::SQUARED_DIFF>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_s32_elementwise_binary<ArithmeticOperation::POWER>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_s32_elementwise_binary<ArithmeticOperation::PRELU>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);

template <ArithmeticOperation op>
void neon_s16_elementwise_binary(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    return elementwise_arithm_op<op, typename wrapper::traits::neon_vector<int16_t, 8>>(in1, in2, out, window);
}
template void neon_s16_elementwise_binary<ArithmeticOperation::ADD>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_s16_elementwise_binary<ArithmeticOperation::SUB>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_s16_elementwise_binary<ArithmeticOperation::DIV>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_s16_elementwise_binary<ArithmeticOperation::MIN>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_s16_elementwise_binary<ArithmeticOperation::MAX>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_s16_elementwise_binary<ArithmeticOperation::SQUARED_DIFF>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_s16_elementwise_binary<ArithmeticOperation::POWER>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_s16_elementwise_binary<ArithmeticOperation::PRELU>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);

template <ComparisonOperation op>
void neon_u8_comparison_elementwise_binary(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    return elementwise_comp_op_8<op, uint8_t, uint8x16_t>(in1, in2, out, window);
}
template void neon_u8_comparison_elementwise_binary<ComparisonOperation::Equal>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_u8_comparison_elementwise_binary<ComparisonOperation::NotEqual>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_u8_comparison_elementwise_binary<ComparisonOperation::Greater>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_u8_comparison_elementwise_binary<ComparisonOperation::GreaterEqual>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_u8_comparison_elementwise_binary<ComparisonOperation::Less>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_u8_comparison_elementwise_binary<ComparisonOperation::LessEqual>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);

template <ComparisonOperation op>
void neon_s16_comparison_elementwise_binary(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    return elementwise_comp_op_16<op, int16_t, int16x8_t>(in1, in2, out, window);
}
template void neon_s16_comparison_elementwise_binary<ComparisonOperation::Equal>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_s16_comparison_elementwise_binary<ComparisonOperation::NotEqual>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_s16_comparison_elementwise_binary<ComparisonOperation::Greater>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_s16_comparison_elementwise_binary<ComparisonOperation::GreaterEqual>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_s16_comparison_elementwise_binary<ComparisonOperation::Less>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_s16_comparison_elementwise_binary<ComparisonOperation::LessEqual>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);

template <ComparisonOperation op>
void neon_s32_comparison_elementwise_binary(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    return elementwise_comp_op_32<op, int32_t, int32x4_t>(in1, in2, out, window);
}
template void neon_s32_comparison_elementwise_binary<ComparisonOperation::Equal>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_s32_comparison_elementwise_binary<ComparisonOperation::NotEqual>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_s32_comparison_elementwise_binary<ComparisonOperation::Greater>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_s32_comparison_elementwise_binary<ComparisonOperation::GreaterEqual>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_s32_comparison_elementwise_binary<ComparisonOperation::Less>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
template void neon_s32_comparison_elementwise_binary<ComparisonOperation::LessEqual>(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window);
}
} // namespace arm_compute
