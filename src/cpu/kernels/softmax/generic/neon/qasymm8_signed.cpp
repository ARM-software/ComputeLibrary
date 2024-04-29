/*
 * Copyright (c) 2021-2024 Arm Limited.
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

#include "src/cpu/kernels/softmax/generic/neon/impl.h"

namespace arm_compute
{
namespace cpu
{
template <bool IS_LOG>
void neon_qasymm8_signed_softmax(
    const ITensor *in, void *const tmp, ITensor *out, const float beta, int axis, const Window &window)
{
    if (axis == 0)
    {
        return neon_softmax_x_quantized<qasymm8_signed_t, IS_LOG>(in, tmp, out, beta, axis, window);
    }
    else
    {
        return neon_softmax_non_x_quantized<qasymm8_signed_t, IS_LOG>(in, tmp, out, beta, axis, window);
    }
}

template void neon_qasymm8_signed_softmax<true>(
    const ITensor *in, void *const tmp, ITensor *out, const float beta, int axis, const Window &window);
template void neon_qasymm8_signed_softmax<false>(
    const ITensor *in, void *const tmp, ITensor *out, const float beta, int axis, const Window &window);

} // namespace cpu
} // namespace arm_compute
