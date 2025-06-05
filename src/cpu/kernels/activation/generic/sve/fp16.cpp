/*
 * Copyright (c) 2020-2025 Arm Limited.
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

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/function_info/ActivationLayerInfo.h"

#include "src/common/utils/profile/acl_profile.h"
#include "src/core/NEON/SVEMath.h"
#include "src/cpu/kernels/lut/list.h"

#include "fp16_impl.h"
#include <arm_sve.h>
#include <cmath>
#include <cstddef>

namespace arm_compute
{
namespace cpu
{
void sve_fp16_activation(const ITensor *src, ITensor *dst, const ActivationLayerInfo &act_info, const Window &window)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "sve_fp16_activation");
    const auto                                    window_start_x = static_cast<int>(window.x().start());
    const auto                                    window_end_x   = static_cast<int>(window.x().end());
    const ActivationLayerInfo::ActivationFunction act            = act_info.activation();

    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(src, win_collapsed);
    Iterator output(dst, win_collapsed);

    dispatch_sve_fp16_activation_function(act, act_info,
                                          [&](auto activation_function)
                                          {
                                              execute_window_loop(
                                                  win_collapsed,
                                                  [&](const Coordinates &)
                                                  {
                                                      const auto input_ptr =
                                                          reinterpret_cast<const float16_t *>(input.ptr());
                                                      const auto output_ptr =
                                                          reinterpret_cast<float16_t *>(output.ptr());

                                                      svfloat16_t tmp;

                                                      int      x  = window_start_x;
                                                      svbool_t pg = svwhilelt_b16(x, window_end_x);
                                                      do
                                                      {
                                                          const auto vin = svld1_f16(pg, input_ptr + x);
                                                          tmp            = activation_function(vin, pg);
                                                          svst1_f16(pg, output_ptr + x, tmp);
                                                          x += svcnth();
                                                          pg = svwhilelt_b16(x, window_end_x);

                                                      } while (svptest_any(svptrue_b16(), pg));
                                                  },
                                                  input, output);
                                          });
}

void sve_fp16_activation_lut(const ITensor             *src,
                             ITensor                   *dst,
                             const ActivationLayerInfo &act_info,
                             const Window              &window)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "sve_fp16_activation_lut");
    ARM_COMPUTE_ERROR_ON(src->info()->data_type() != DataType::F16);
    const auto window_start_x = window.x().start();
    const auto window_end_x   = window.x().end();
    const auto size           = window_end_x - window_start_x;
    Window     win_collapsed  = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(src, win_collapsed);
    Iterator output(dst, win_collapsed);
    execute_window_loop(
        win_collapsed,
        [&](const Coordinates &)
        {
            const auto input_ptr  = reinterpret_cast<const uint16_t *>(input.ptr());
            auto       output_ptr = reinterpret_cast<uint16_t *>(output.ptr());
            lut_u16_sve(reinterpret_cast<const uint16_t *>(act_info.lut_fp16().data()), 1U /* num_strings (UNUSED) */,
                        size, input_ptr + window_start_x, output_ptr + window_start_x);
        },
        input, output);
}
} // namespace cpu
} // namespace arm_compute
#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS) */
