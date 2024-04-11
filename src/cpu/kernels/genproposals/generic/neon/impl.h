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
#ifndef SRC_CORE_SVE_KERNELS_NEGENERATEPROPOSALSLAYERKERNEL_IMPL_H
#define SRC_CORE_SVE_KERNELS_NEGENERATEPROPOSALSLAYERKERNEL_IMPL_H
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Window.h"

#include "src/core/NEON/wrapper/wrapper.h"
namespace arm_compute
{
namespace cpu
{
template <typename T>
void compute_all_anchors(const ITensor     *anchors,
                         ITensor           *all_anchors,
                         ComputeAnchorsInfo anchors_info,
                         const Window      &window)
{
    Iterator all_anchors_it(all_anchors, window);
    Iterator anchors_it(all_anchors, window);

    const size_t num_anchors = anchors->info()->dimension(1);
    const T      stride      = 1.f / anchors_info.spatial_scale();
    const size_t feat_width  = anchors_info.feat_width();

    execute_window_loop(
        window,
        [&](const Coordinates &id)
        {
            const size_t anchor_offset = id.y() % num_anchors;

            const auto out_anchor_ptr = reinterpret_cast<T *>(all_anchors_it.ptr());
            const auto anchor_ptr     = reinterpret_cast<T *>(anchors->ptr_to_element(Coordinates(0, anchor_offset)));

            const size_t shift_idy = id.y() / num_anchors;
            const T      shiftx    = (shift_idy % feat_width) * stride;
            const T      shifty    = (shift_idy / feat_width) * stride;

            *out_anchor_ptr       = *anchor_ptr + shiftx;
            *(out_anchor_ptr + 1) = *(1 + anchor_ptr) + shifty;
            *(out_anchor_ptr + 2) = *(2 + anchor_ptr) + shiftx;
            *(out_anchor_ptr + 3) = *(3 + anchor_ptr) + shifty;
        },
        all_anchors_it);
}

void compute_all_anchors_qasymm16(const ITensor     *anchors,
                                  ITensor           *all_anchors,
                                  ComputeAnchorsInfo anchors_info,
                                  const Window      &window);
} // namespace cpu
} // namespace arm_compute
#endif //define SRC_CORE_SVE_KERNELS_NEGENERATEPROPOSALSLAYERKERNEL_IMPL_H
