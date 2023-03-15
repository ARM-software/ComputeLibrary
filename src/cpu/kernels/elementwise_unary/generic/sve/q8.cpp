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

#include "arm_compute/core/Helpers.h"
#include "src/cpu/kernels/lut/list.h"

namespace arm_compute
{
namespace cpu
{

void sve_q8_elementwise_unary(const ITensor *in, ITensor *out, const Window &window, ElementWiseUnary op, const uint8_t *lut)
{
    ARM_COMPUTE_UNUSED(op);

    auto win = window;
    const auto window_end_x = window.x().end();
    win.set(0, Window::Dimension(0, 1, 1));

    Iterator src_it(in, win);
    Iterator dst_it(out, win);

    execute_window_loop(win, [&](const Coordinates &) {
        const auto src_ptr = src_it.ptr();
        auto dst_ptr = dst_it.ptr();

        lut_u8_sve(lut, 1, window_end_x, &src_ptr, &dst_ptr);
    },
    src_it, dst_it);
}

} // namespace cpu
} // namespace arm_compute
