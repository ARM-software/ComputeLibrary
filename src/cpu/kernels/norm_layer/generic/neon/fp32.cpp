/*
 * Copyright (c) 2023 Arm Limited.
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

#include "src/core/NEON/wrapper/wrapper.h"
#include "src/cpu/CpuTypes.h"
#include "src/cpu/kernels/norm_layer/generic/neon/impl.h"

namespace arm_compute
{
namespace cpu
{
void neon_normalize_float32_4_0_2D(
    const Window &window, const ITensor *in, const ITensor *in_squared, ITensor *out, NormalizationLayerInfo ninfo)
{
    arm_compute::normalize_float<float, 4, 0, true>(window, in, in_squared, out, ninfo);
}

void neon_normalize_float32_4_0(
    const Window &window, const ITensor *in, const ITensor *in_squared, ITensor *out, NormalizationLayerInfo ninfo)
{
    arm_compute::normalize_float<float, 4, 0, false>(window, in, in_squared, out, ninfo);
}

void neon_normalize_float32_4_1_2D(
    const Window &window, const ITensor *in, const ITensor *in_squared, ITensor *out, NormalizationLayerInfo ninfo)
{
    arm_compute::normalize_float<float, 4, 1, true>(window, in, in_squared, out, ninfo);
}

void neon_normalize_float32_4_1(
    const Window &window, const ITensor *in, const ITensor *in_squared, ITensor *out, NormalizationLayerInfo ninfo)
{
    arm_compute::normalize_float<float, 4, 1, false>(window, in, in_squared, out, ninfo);
}

void neon_normalize_float32_4_2(
    const Window &window, const ITensor *in, const ITensor *in_squared, ITensor *out, NormalizationLayerInfo ninfo)
{
    arm_compute::normalize_float<float, 4, 2, false>(window, in, in_squared, out, ninfo);
}
} // namespace cpu
} // namespace arm_compute
