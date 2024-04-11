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
#ifndef SRC_CORE_SVE_KERNELS_BOUNDINGBOXTRANFORM_IMPL_H
#define SRC_CORE_SVE_KERNELS_BOUNDINGBOXTRANFORM_IMPL_H
#include "arm_compute/core/Helpers.h"

namespace arm_compute
{
namespace cpu
{
template <typename T>
void bounding_box_transform(const ITensor           *boxes,
                            ITensor                 *pred_boxes,
                            const ITensor           *deltas,
                            BoundingBoxTransformInfo bbinfo,
                            const Window            &window)
{
    const size_t num_classes  = deltas->info()->tensor_shape()[0] >> 2;
    const size_t deltas_width = deltas->info()->tensor_shape()[0];
    const int    img_h        = std::floor(bbinfo.img_height() / bbinfo.scale() + 0.5f);
    const int    img_w        = std::floor(bbinfo.img_width() / bbinfo.scale() + 0.5f);

    const auto scale_after  = (bbinfo.apply_scale() ? T(bbinfo.scale()) : T(1));
    const auto scale_before = T(bbinfo.scale());
    ARM_COMPUTE_ERROR_ON(scale_before <= 0);
    const auto offset = (bbinfo.correct_transform_coords() ? T(1.f) : T(0.f));

    auto pred_ptr  = reinterpret_cast<T *>(pred_boxes->buffer() + pred_boxes->info()->offset_first_element_in_bytes());
    auto delta_ptr = reinterpret_cast<T *>(deltas->buffer() + deltas->info()->offset_first_element_in_bytes());

    Iterator box_it(boxes, window);
    execute_window_loop(
        window,
        [&](const Coordinates &id)
        {
            const auto ptr    = reinterpret_cast<T *>(box_it.ptr());
            const auto b0     = *ptr;
            const auto b1     = *(ptr + 1);
            const auto b2     = *(ptr + 2);
            const auto b3     = *(ptr + 3);
            const T    width  = (b2 / scale_before) - (b0 / scale_before) + T(1.f);
            const T    height = (b3 / scale_before) - (b1 / scale_before) + T(1.f);
            const T    ctr_x  = (b0 / scale_before) + T(0.5f) * width;
            const T    ctr_y  = (b1 / scale_before) + T(0.5f) * height;
            for (size_t j = 0; j < num_classes; ++j)
            {
                // Extract deltas
                const size_t delta_id = id.y() * deltas_width + 4u * j;
                const T      dx       = delta_ptr[delta_id] / T(bbinfo.weights()[0]);
                const T      dy       = delta_ptr[delta_id + 1] / T(bbinfo.weights()[1]);
                T            dw       = delta_ptr[delta_id + 2] / T(bbinfo.weights()[2]);
                T            dh       = delta_ptr[delta_id + 3] / T(bbinfo.weights()[3]);
                // Clip dw and dh
                dw = std::min(dw, T(bbinfo.bbox_xform_clip()));
                dh = std::min(dh, T(bbinfo.bbox_xform_clip()));
                // Determine the predictions
                const T pred_ctr_x = dx * width + ctr_x;
                const T pred_ctr_y = dy * height + ctr_y;
                const T pred_w     = std::exp(dw) * width;
                const T pred_h     = std::exp(dh) * height;
                // Store the prediction into the output tensor
                pred_ptr[delta_id] = scale_after * utility::clamp<T>(pred_ctr_x - T(0.5f) * pred_w, T(0), T(img_w - 1));
                pred_ptr[delta_id + 1] =
                    scale_after * utility::clamp<T>(pred_ctr_y - T(0.5f) * pred_h, T(0), T(img_h - 1));
                pred_ptr[delta_id + 2] =
                    scale_after * utility::clamp<T>(pred_ctr_x + T(0.5f) * pred_w - offset, T(0), T(img_w - 1));
                pred_ptr[delta_id + 3] =
                    scale_after * utility::clamp<T>(pred_ctr_y + T(0.5f) * pred_h - offset, T(0), T(img_h - 1));
            }
        },
        box_it);
}

void bounding_box_transform_qsymm16(const ITensor           *boxes,
                                    ITensor                 *pred_boxes,
                                    const ITensor           *deltas,
                                    BoundingBoxTransformInfo bbinfo,
                                    const Window            &window);
} // namespace cpu
} // namespace arm_compute
#endif //define SRC_CORE_SVE_KERNELS_BOUNDINGBOXTRANFORM_IMPL_H
