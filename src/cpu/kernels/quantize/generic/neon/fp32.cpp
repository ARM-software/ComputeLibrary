/*
 * Copyright (c) 2024-2025 Arm Limited.
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
#include "src/cpu/kernels/quantize/generic/neon/impl.h"

namespace arm_compute
{
namespace cpu
{
void fp32_u8_run_quantize_qasymm8(const ITensor *src, ITensor *dst, const Window &window)
{
    run_quantize_qasymm8<float, uint8_t>(src, dst, window);
}
void fp32_i8_run_quantize_qasymm8(const ITensor *src, ITensor *dst, const Window &window)
{
    run_quantize_qasymm8<float, int8_t>(src, dst, window);
}
void fp32_run_quantize_qasymm16(const ITensor *src, ITensor *dst, const Window &window)
{
    run_quantize_qasymm16<float>(src, dst, window);
}

void fp32_i8_run_quantize_qsymm8(const ITensor *src, ITensor *dst, const Window &window)
{
    run_quantize_qsymm8<float, int8_t>(src, dst, window);
}

void fp32_i8_run_quantize_qsymm8_per_channel(const ITensor *src, ITensor *dst, const Window &window)
{
    run_quantize_qsymm8_per_channel<float, int8_t>(src, dst, window);
}
} // namespace cpu
} // namespace arm_compute
