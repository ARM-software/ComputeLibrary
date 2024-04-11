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
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"

#include "src/cpu/kernels/select/generic/neon/impl.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace cpu
{
void neon_s8_select_same_rank(
    const ITensor *c, const ITensor *x, const ITensor *y, ITensor *output, const Window &window)
{
    return select_op_8<int8_t, uint8x16_t>(c, x, y, output, window);
}
void neon_s16_select_same_rank(
    const ITensor *c, const ITensor *x, const ITensor *y, ITensor *output, const Window &window)
{
    return select_op_16<int16_t, uint16x8_t>(c, x, y, output, window);
}
void neon_s32_select_same_rank(
    const ITensor *c, const ITensor *x, const ITensor *y, ITensor *output, const Window &window)
{
    return select_op_32<int32_t, uint32x4_t>(c, x, y, output, window);
}
void neon_s8_select_not_same_rank(
    const ITensor *c, const ITensor *x, const ITensor *y, ITensor *output, const Window &window)
{
    return select_op_not_same_rank<int8_t>(c, x, y, output, window);
}
void neon_s16_select_not_same_rank(
    const ITensor *c, const ITensor *x, const ITensor *y, ITensor *output, const Window &window)
{
    return select_op_not_same_rank<int16_t>(c, x, y, output, window);
}
void neon_s32_select_not_same_rank(
    const ITensor *c, const ITensor *x, const ITensor *y, ITensor *output, const Window &window)
{
    return select_op_not_same_rank<int32_t>(c, x, y, output, window);
}
void neon_u8_select_same_rank(
    const ITensor *c, const ITensor *x, const ITensor *y, ITensor *output, const Window &window)
{
    return select_op_8<uint8_t, uint8x16_t>(c, x, y, output, window);
}
void neon_u16_select_same_rank(
    const ITensor *c, const ITensor *x, const ITensor *y, ITensor *output, const Window &window)
{
    return select_op_16<uint16_t, uint16x8_t>(c, x, y, output, window);
}
void neon_u32_select_same_rank(
    const ITensor *c, const ITensor *x, const ITensor *y, ITensor *output, const Window &window)
{
    return select_op_32<uint32_t, uint32x4_t>(c, x, y, output, window);
}
void neon_u8_select_not_same_rank(
    const ITensor *c, const ITensor *x, const ITensor *y, ITensor *output, const Window &window)
{
    return select_op_not_same_rank<uint8_t>(c, x, y, output, window);
}
void neon_u16_select_not_same_rank(
    const ITensor *c, const ITensor *x, const ITensor *y, ITensor *output, const Window &window)
{
    return select_op_not_same_rank<uint16_t>(c, x, y, output, window);
}
void neon_u32_select_not_same_rank(
    const ITensor *c, const ITensor *x, const ITensor *y, ITensor *output, const Window &window)
{
    return select_op_not_same_rank<uint32_t>(c, x, y, output, window);
}

} // namespace cpu

} // namespace arm_compute
