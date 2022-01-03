/*
 * Copyright (c) 2019-2022 Arm Limited.
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
#include "src/cpu/kernels/boundingboxtransform/generic/neon/impl.h"
namespace arm_compute
{
namespace cpu
{
void bounding_box_transform_qsymm16(const ITensor *boxes, ITensor *pred_boxes, const ITensor *deltas, BoundingBoxTransformInfo bbinfo, const Window &window)

{
    const size_t num_classes  = deltas->info()->tensor_shape()[0] >> 2;
    const size_t deltas_width = deltas->info()->tensor_shape()[0];
    const int    img_h        = std::floor(bbinfo.img_height() / bbinfo.scale() + 0.5f);
    const int    img_w        = std::floor(bbinfo.img_width() / bbinfo.scale() + 0.5f);

    const auto scale_after  = (bbinfo.apply_scale() ? bbinfo.scale() : 1.f);
    const auto scale_before = bbinfo.scale();
    const auto offset       = (bbinfo.correct_transform_coords() ? 1.f : 0.f);

    auto pred_ptr  = reinterpret_cast<uint16_t *>(pred_boxes->buffer() + pred_boxes->info()->offset_first_element_in_bytes());
    auto delta_ptr = reinterpret_cast<uint8_t *>(deltas->buffer() + deltas->info()->offset_first_element_in_bytes());

    const auto boxes_qinfo  = boxes->info()->quantization_info().uniform();
    const auto deltas_qinfo = deltas->info()->quantization_info().uniform();
    const auto pred_qinfo   = pred_boxes->info()->quantization_info().uniform();

    Iterator box_it(boxes, window);
    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto  ptr    = reinterpret_cast<uint16_t *>(box_it.ptr());
        const auto  b0     = dequantize_qasymm16(*ptr, boxes_qinfo);
        const auto  b1     = dequantize_qasymm16(*(ptr + 1), boxes_qinfo);
        const auto  b2     = dequantize_qasymm16(*(ptr + 2), boxes_qinfo);
        const auto  b3     = dequantize_qasymm16(*(ptr + 3), boxes_qinfo);
        const float width  = (b2 / scale_before) - (b0 / scale_before) + 1.f;
        const float height = (b3 / scale_before) - (b1 / scale_before) + 1.f;
        const float ctr_x  = (b0 / scale_before) + 0.5f * width;
        const float ctr_y  = (b1 / scale_before) + 0.5f * height;
        for(size_t j = 0; j < num_classes; ++j)
        {
            // Extract deltas
            const size_t delta_id = id.y() * deltas_width + 4u * j;
            const float  dx       = dequantize_qasymm8(delta_ptr[delta_id], deltas_qinfo) / bbinfo.weights()[0];
            const float  dy       = dequantize_qasymm8(delta_ptr[delta_id + 1], deltas_qinfo) / bbinfo.weights()[1];
            float        dw       = dequantize_qasymm8(delta_ptr[delta_id + 2], deltas_qinfo) / bbinfo.weights()[2];
            float        dh       = dequantize_qasymm8(delta_ptr[delta_id + 3], deltas_qinfo) / bbinfo.weights()[3];
            // Clip dw and dh
            dw = std::min(dw, bbinfo.bbox_xform_clip());
            dh = std::min(dh, bbinfo.bbox_xform_clip());
            // Determine the predictions
            const float pred_ctr_x = dx * width + ctr_x;
            const float pred_ctr_y = dy * height + ctr_y;
            const float pred_w     = std::exp(dw) * width;
            const float pred_h     = std::exp(dh) * height;
            // Store the prediction into the output tensor
            pred_ptr[delta_id]     = quantize_qasymm16(scale_after * utility::clamp<float>(pred_ctr_x - 0.5f * pred_w, 0.f, img_w - 1.f), pred_qinfo);
            pred_ptr[delta_id + 1] = quantize_qasymm16(scale_after * utility::clamp<float>(pred_ctr_y - 0.5f * pred_h, 0.f, img_h - 1.f), pred_qinfo);
            pred_ptr[delta_id + 2] = quantize_qasymm16(scale_after * utility::clamp<float>(pred_ctr_x + 0.5f * pred_w - offset, 0.f, img_w - 1.f), pred_qinfo);
            pred_ptr[delta_id + 3] = quantize_qasymm16(scale_after * utility::clamp<float>(pred_ctr_y + 0.5f * pred_h - offset, 0.f, img_h - 1.f), pred_qinfo);
        }
    },
    box_it);
}

template <typename T>
void bounding_box_transform(const ITensor *boxes, ITensor *pred_boxes, const ITensor *deltas, BoundingBoxTransformInfo bbinfo, const Window &window)
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
    execute_window_loop(window, [&](const Coordinates & id)
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
        for(size_t j = 0; j < num_classes; ++j)
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
            pred_ptr[delta_id]     = scale_after * utility::clamp<T>(pred_ctr_x - T(0.5f) * pred_w, T(0), T(img_w - 1));
            pred_ptr[delta_id + 1] = scale_after * utility::clamp<T>(pred_ctr_y - T(0.5f) * pred_h, T(0), T(img_h - 1));
            pred_ptr[delta_id + 2] = scale_after * utility::clamp<T>(pred_ctr_x + T(0.5f) * pred_w - offset, T(0), T(img_w - 1));
            pred_ptr[delta_id + 3] = scale_after * utility::clamp<T>(pred_ctr_y + T(0.5f) * pred_h - offset, T(0), T(img_h - 1));
        }
    },
    box_it);
}

template void bounding_box_transform<float>(const ITensor *boxes, ITensor *pred_boxes, const ITensor *deltas, BoundingBoxTransformInfo bbinfo, const Window &window);

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)
template void bounding_box_transform<float16_t>(const ITensor *boxes, ITensor *pred_boxes, const ITensor *deltas, BoundingBoxTransformInfo bbinfo, const Window &window);
#endif //defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)
} // namespace cpu
} // namespace arm_compute