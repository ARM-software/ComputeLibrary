/*
 * Copyright (c) 2021 Arm Limited.
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

#include "src/cpu/kernels/range/generic/neon/impl.h"

#include <cstdint>

namespace arm_compute
{
namespace cpu
{
void u8_neon_range_function(ITensor *output, float start, float step, const Window &window)
{
    return neon_range_function<uint8_t>(output, start, step, window);
}

void u16_neon_range_function(ITensor *output, float start, float step, const Window &window)
{
    return neon_range_function<uint16_t>(output, start, step, window);
}

void u32_neon_range_function(ITensor *output, float start, float step, const Window &window)
{
    return neon_range_function<uint32_t>(output, start, step, window);
}

void s8_neon_range_function(ITensor *output, float start, float step, const Window &window)
{
    return neon_range_function<int8_t>(output, start, step, window);
}

void s16_neon_range_function(ITensor *output, float start, float step, const Window &window)
{
    return neon_range_function<int16_t>(output, start, step, window);
}

void s32_neon_range_function(ITensor *output, float start, float step, const Window &window)
{
    return neon_range_function<int32_t>(output, start, step, window);
}

} // namespace cpu
} // namespace arm_compute
