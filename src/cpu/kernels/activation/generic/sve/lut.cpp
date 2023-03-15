/*
 * Copyright (c) 2022-2023 Arm Limited.
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
#ifdef __aarch64__
void sve_q8_activation_lut(const ITensor *src, ITensor *dst, const ActivationLayerInfo &act_info, const Window &window)
{
    ARM_COMPUTE_ERROR_ON(!ActivationLayerInfo::is_lut_supported(act_info.activation(), src->info()->data_type()));
    const auto window_end_x  = window.x().end();
    Window     win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));
    Iterator input(src, win_collapsed);
    Iterator output(dst, win_collapsed);
    execute_window_loop(win_collapsed, [&](const Coordinates &)
    {
        const auto input_ptr  = input.ptr();
        auto       output_ptr = output.ptr();
        lut_u8_sve(act_info.lut().data(), 1u, window_end_x, &input_ptr, &output_ptr);
    },
    input, output);
}
#endif // __aarch64__
} // namespace cpu
} // namespace arm_compute
