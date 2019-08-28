/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "BoundingBoxTransform.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/misc/Utility.h"
#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T, typename TDeltas>
SimpleTensor<T> bounding_box_transform(const SimpleTensor<T> &boxes, const SimpleTensor<TDeltas> &deltas, const BoundingBoxTransformInfo &info)
{
    const DataType  boxes_data_type = boxes.data_type();
    SimpleTensor<T> pred_boxes(deltas.shape(), boxes_data_type);

    const size_t   num_classes    = deltas.shape()[0] / 4;
    const size_t   num_boxes      = deltas.shape()[1];
    const TDeltas *deltas_ptr     = deltas.data();
    T             *pred_boxes_ptr = pred_boxes.data();

    const int img_h = floor(info.img_height() / info.scale() + 0.5f);
    const int img_w = floor(info.img_width() / info.scale() + 0.5f);

    const auto scale_after  = (info.apply_scale() ? T(info.scale()) : T(1));
    const auto scale_before = T(info.scale());
    ARM_COMPUTE_ERROR_ON(scale_before <= 0);
    const auto offset = (info.correct_transform_coords() ? T(1.f) : T(0.f));

    const size_t box_fields   = 4;
    const size_t class_fields = 4;

    for(size_t i = 0; i < num_boxes; ++i)
    {
        // Extract ROI information
        const size_t start_box = box_fields * i;
        const T      width     = (boxes[start_box + 2] / scale_before) - (boxes[start_box] / scale_before) + T(1.f);
        const T      height    = (boxes[start_box + 3] / scale_before) - (boxes[start_box + 1] / scale_before) + T(1.f);
        const T      ctr_x     = (boxes[start_box] / scale_before) + T(0.5f) * width;
        const T      ctr_y     = (boxes[start_box + 1] / scale_before) + T(0.5f) * height;

        for(size_t j = 0; j < num_classes; ++j)
        {
            // Extract deltas
            const size_t  start_delta = i * num_classes * class_fields + class_fields * j;
            const TDeltas dx          = deltas_ptr[start_delta] / TDeltas(info.weights()[0]);
            const TDeltas dy          = deltas_ptr[start_delta + 1] / TDeltas(info.weights()[1]);
            TDeltas       dw          = deltas_ptr[start_delta + 2] / TDeltas(info.weights()[2]);
            TDeltas       dh          = deltas_ptr[start_delta + 3] / TDeltas(info.weights()[3]);

            // Clip dw and dh
            dw = std::min(dw, TDeltas(info.bbox_xform_clip()));
            dh = std::min(dh, TDeltas(info.bbox_xform_clip()));

            // Determine the predictions
            const T pred_ctr_x = dx * width + ctr_x;
            const T pred_ctr_y = dy * height + ctr_y;
            const T pred_w     = T(std::exp(dw)) * width;
            const T pred_h     = T(std::exp(dh)) * height;

            // Store the prediction into the output tensor
            pred_boxes_ptr[start_delta]     = scale_after * utility::clamp<T>(pred_ctr_x - T(0.5f) * pred_w, T(0), T(img_w - 1));
            pred_boxes_ptr[start_delta + 1] = scale_after * utility::clamp<T>(pred_ctr_y - T(0.5f) * pred_h, T(0), T(img_h - 1));
            pred_boxes_ptr[start_delta + 2] = scale_after * utility::clamp<T>(pred_ctr_x + T(0.5f) * pred_w - offset, T(0), T(img_w - 1));
            pred_boxes_ptr[start_delta + 3] = scale_after * utility::clamp<T>(pred_ctr_y + T(0.5f) * pred_h - offset, T(0), T(img_h - 1));
        }
    }
    return pred_boxes;
}

template SimpleTensor<float> bounding_box_transform(const SimpleTensor<float> &boxes, const SimpleTensor<float> &deltas, const BoundingBoxTransformInfo &info);
template SimpleTensor<half> bounding_box_transform(const SimpleTensor<half> &boxes, const SimpleTensor<half> &deltas, const BoundingBoxTransformInfo &info);

template <>
SimpleTensor<uint16_t> bounding_box_transform(const SimpleTensor<uint16_t> &boxes, const SimpleTensor<uint8_t> &deltas, const BoundingBoxTransformInfo &info)
{
    SimpleTensor<float>    boxes_tmp      = convert_from_asymmetric(boxes);
    SimpleTensor<float>    deltas_tmp     = convert_from_asymmetric(deltas);
    SimpleTensor<float>    pred_boxes_tmp = bounding_box_transform<float, float>(boxes_tmp, deltas_tmp, info);
    SimpleTensor<uint16_t> pred_boxes     = convert_to_asymmetric<uint16_t>(pred_boxes_tmp, boxes.quantization_info());
    return pred_boxes;
}
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
