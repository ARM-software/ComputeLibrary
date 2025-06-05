/*
 * Copyright (c) 2020-2023, 2025 Arm Limited.
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
#include "arm_compute/core/experimental/Types.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/function_info/ActivationLayerInfo.h"

#include "src/common/utils/profile/acl_profile.h"

#include "qsymm16_impl.h"
#include <arm_sve.h>
#include <cmath>
#include <cstddef>

namespace arm_compute
{
namespace cpu
{
void sve2_qsymm16_activation(const ITensor             *src,
                             ITensor                   *dst,
                             const ActivationLayerInfo &act_info,
                             const Window              &window)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "sve2_qsymm16_activation");
    const auto                                    window_start_x = static_cast<int>(window.x().start());
    const auto                                    window_end_x   = static_cast<int>(window.x().end());
    const ActivationLayerInfo::ActivationFunction act            = act_info.activation();

    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(src, win_collapsed);
    Iterator output(dst, win_collapsed);

    const UniformQuantizationInfo qi_in  = src->info()->quantization_info().uniform();
    const UniformQuantizationInfo qi_out = dst->info()->quantization_info().uniform();

    dispatch_sve2_qasymm16_activation_function(act, act_info, qi_in, qi_out,
                                               [&](auto activation_function)
                                               {
                                                   execute_window_loop(
                                                       win_collapsed,
                                                       [&](const Coordinates &)
                                                       {
                                                           const auto input_ptr =
                                                               reinterpret_cast<const int16_t *>(input.ptr());
                                                           const auto output_ptr =
                                                               reinterpret_cast<int16_t *>(output.ptr());

                                                           svint16_t tmp;

                                                           int      x  = window_start_x;
                                                           svbool_t pg = svwhilelt_b16(x, window_end_x);
                                                           do
                                                           {
                                                               const auto vin = svld1_s16(pg, input_ptr + x);
                                                               tmp            = activation_function(vin, pg);
                                                               svst1_s16(pg, output_ptr + x, tmp);

                                                               x += svcnth();
                                                               pg = svwhilelt_b16(x, window_end_x);

                                                           } while (svptest_any(svptrue_b16(), pg));
                                                       },
                                                       input, output);
                                               });
}
} // namespace cpu
} // namespace arm_compute
