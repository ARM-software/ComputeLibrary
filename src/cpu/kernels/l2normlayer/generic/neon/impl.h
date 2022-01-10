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
#ifndef SRC_CORE_NEON_KERNELS_L2NORMLAYER_LIST_H
#define SRC_CORE_NEON_KERNELS_L2NORMLAYER_LIST_H

#include <cstddef>

namespace arm_compute
{
class ITensor;
class Window;

namespace cpu
{
template <typename T, int S>
void l2_normalize_x(const ITensor *in, const ITensor *sum, ITensor *out, float epsilon, const Window &window);

template <typename T, int S>
void l2_normalize_yz(const ITensor *in, const ITensor *sum, ITensor *out, float epsilon, const Window &window, size_t axis);

} // namespace cpu
} // namespace arm_compute
#endif //SRC_CORE_NEON_KERNELS_L2NORMLAYER_LIST_H
