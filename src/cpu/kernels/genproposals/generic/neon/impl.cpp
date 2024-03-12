/*
 * Copyright (c) 2019-2023 Arm Limited.
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
#include "src/cpu/kernels/genproposals/generic/neon/impl.h"
namespace arm_compute
{
class ITensor;
class Window;
namespace cpu
{
void compute_all_anchors_qasymm16(const ITensor     *anchors,
                                  ITensor           *all_anchors,
                                  ComputeAnchorsInfo anchors_info,
                                  const Window      &window)
{
    Iterator all_anchors_it(all_anchors, window);
    Iterator anchors_it(all_anchors, window);

    const size_t num_anchors = anchors->info()->dimension(1);
    const float  stride      = 1.f / anchors_info.spatial_scale();
    const size_t feat_width  = anchors_info.feat_width();

    const UniformQuantizationInfo qinfo = anchors->info()->quantization_info().uniform();

    execute_window_loop(
        window,
        [&](const Coordinates &id)
        {
            const size_t anchor_offset = id.y() % num_anchors;

            const auto out_anchor_ptr = reinterpret_cast<int16_t *>(all_anchors_it.ptr());
            const auto anchor_ptr = reinterpret_cast<int16_t *>(anchors->ptr_to_element(Coordinates(0, anchor_offset)));

            const size_t shift_idy = id.y() / num_anchors;
            const float  shiftx    = (shift_idy % feat_width) * stride;
            const float  shifty    = (shift_idy / feat_width) * stride;

            const float new_anchor_x1 = dequantize_qsymm16(*anchor_ptr, qinfo.scale) + shiftx;
            const float new_anchor_y1 = dequantize_qsymm16(*(1 + anchor_ptr), qinfo.scale) + shifty;
            const float new_anchor_x2 = dequantize_qsymm16(*(2 + anchor_ptr), qinfo.scale) + shiftx;
            const float new_anchor_y2 = dequantize_qsymm16(*(3 + anchor_ptr), qinfo.scale) + shifty;

            *out_anchor_ptr       = quantize_qsymm16(new_anchor_x1, qinfo.scale);
            *(out_anchor_ptr + 1) = quantize_qsymm16(new_anchor_y1, qinfo.scale);
            *(out_anchor_ptr + 2) = quantize_qsymm16(new_anchor_x2, qinfo.scale);
            *(out_anchor_ptr + 3) = quantize_qsymm16(new_anchor_y2, qinfo.scale);
        },
        all_anchors_it);
}
} // namespace cpu
} // namespace arm_compute
