/*
 * Copyright (c) 2019-2020 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEBoundingBoxTransformKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *boxes, const ITensorInfo *pred_boxes, const ITensorInfo *deltas, const BoundingBoxTransformInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(boxes, pred_boxes, deltas);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(boxes);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(boxes, DataType::QASYMM16, DataType::F32, DataType::F16);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(deltas, DataType::QASYMM8, DataType::F32, DataType::F16);
    ARM_COMPUTE_RETURN_ERROR_ON(deltas->tensor_shape()[1] != boxes->tensor_shape()[1]);
    ARM_COMPUTE_RETURN_ERROR_ON(deltas->tensor_shape()[0] % 4 != 0);
    ARM_COMPUTE_RETURN_ERROR_ON(boxes->tensor_shape()[0] != 4);
    ARM_COMPUTE_RETURN_ERROR_ON(deltas->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON(boxes->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON(info.scale() <= 0);

    if(boxes->data_type() == DataType::QASYMM16)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(deltas, 1, DataType::QASYMM8);
        const UniformQuantizationInfo deltas_qinfo = deltas->quantization_info().uniform();
        ARM_COMPUTE_RETURN_ERROR_ON(deltas_qinfo.scale != 0.125f);
        ARM_COMPUTE_RETURN_ERROR_ON(deltas_qinfo.offset != 0);
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(boxes, deltas);
    }

    if(pred_boxes->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(pred_boxes->tensor_shape(), deltas->tensor_shape());
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(pred_boxes, deltas);
        ARM_COMPUTE_RETURN_ERROR_ON(pred_boxes->num_dimensions() > 2);
        if(pred_boxes->data_type() == DataType::QASYMM16)
        {
            const UniformQuantizationInfo pred_qinfo = pred_boxes->quantization_info().uniform();
            ARM_COMPUTE_RETURN_ERROR_ON(pred_qinfo.scale != 0.125f);
            ARM_COMPUTE_RETURN_ERROR_ON(pred_qinfo.offset != 0);
        }
    }

    return Status{};
}
} // namespace

NEBoundingBoxTransformKernel::NEBoundingBoxTransformKernel()
    : _boxes(nullptr), _pred_boxes(nullptr), _deltas(nullptr), _bbinfo(0, 0, 0)
{
}

void NEBoundingBoxTransformKernel::configure(const ITensor *boxes, ITensor *pred_boxes, const ITensor *deltas, const BoundingBoxTransformInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(boxes, pred_boxes, deltas);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(boxes->info(), pred_boxes->info(), deltas->info(), info));

    // Configure kernel window
    auto_init_if_empty(*pred_boxes->info(), deltas->info()->clone()->set_data_type(boxes->info()->data_type()).set_quantization_info(boxes->info()->quantization_info()));

    // Set instance variables
    _boxes      = boxes;
    _pred_boxes = pred_boxes;
    _deltas     = deltas;
    _bbinfo     = info;

    const unsigned int num_boxes = boxes->info()->dimension(1);
    Window             win       = calculate_max_window(*pred_boxes->info(), Steps());
    Coordinates        coord;
    coord.set_num_dimensions(pred_boxes->info()->num_dimensions());
    pred_boxes->info()->set_valid_region(ValidRegion(coord, pred_boxes->info()->tensor_shape()));
    win.set(Window::DimX, Window::Dimension(0, 1u));
    win.set(Window::DimY, Window::Dimension(0, num_boxes));

    INEKernel::configure(win);
}

Status NEBoundingBoxTransformKernel::validate(const ITensorInfo *boxes, const ITensorInfo *pred_boxes, const ITensorInfo *deltas, const BoundingBoxTransformInfo &info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(boxes, pred_boxes, deltas, info));
    return Status{};
}

template <>
void NEBoundingBoxTransformKernel::internal_run<uint16_t>(const Window &window)
{
    const size_t num_classes  = _deltas->info()->tensor_shape()[0] >> 2;
    const size_t deltas_width = _deltas->info()->tensor_shape()[0];
    const int    img_h        = std::floor(_bbinfo.img_height() / _bbinfo.scale() + 0.5f);
    const int    img_w        = std::floor(_bbinfo.img_width() / _bbinfo.scale() + 0.5f);

    const auto scale_after  = (_bbinfo.apply_scale() ? _bbinfo.scale() : 1.f);
    const auto scale_before = _bbinfo.scale();
    const auto offset       = (_bbinfo.correct_transform_coords() ? 1.f : 0.f);

    auto pred_ptr  = reinterpret_cast<uint16_t *>(_pred_boxes->buffer() + _pred_boxes->info()->offset_first_element_in_bytes());
    auto delta_ptr = reinterpret_cast<uint8_t *>(_deltas->buffer() + _deltas->info()->offset_first_element_in_bytes());

    const auto boxes_qinfo  = _boxes->info()->quantization_info().uniform();
    const auto deltas_qinfo = _deltas->info()->quantization_info().uniform();
    const auto pred_qinfo   = _pred_boxes->info()->quantization_info().uniform();

    Iterator box_it(_boxes, window);
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
            const float  dx       = dequantize_qasymm8(delta_ptr[delta_id], deltas_qinfo) / _bbinfo.weights()[0];
            const float  dy       = dequantize_qasymm8(delta_ptr[delta_id + 1], deltas_qinfo) / _bbinfo.weights()[1];
            float        dw       = dequantize_qasymm8(delta_ptr[delta_id + 2], deltas_qinfo) / _bbinfo.weights()[2];
            float        dh       = dequantize_qasymm8(delta_ptr[delta_id + 3], deltas_qinfo) / _bbinfo.weights()[3];
            // Clip dw and dh
            dw = std::min(dw, _bbinfo.bbox_xform_clip());
            dh = std::min(dh, _bbinfo.bbox_xform_clip());
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
void NEBoundingBoxTransformKernel::internal_run(const Window &window)
{
    const size_t num_classes  = _deltas->info()->tensor_shape()[0] >> 2;
    const size_t deltas_width = _deltas->info()->tensor_shape()[0];
    const int    img_h        = std::floor(_bbinfo.img_height() / _bbinfo.scale() + 0.5f);
    const int    img_w        = std::floor(_bbinfo.img_width() / _bbinfo.scale() + 0.5f);

    const auto scale_after  = (_bbinfo.apply_scale() ? T(_bbinfo.scale()) : T(1));
    const auto scale_before = T(_bbinfo.scale());
    ARM_COMPUTE_ERROR_ON(scale_before <= 0);
    const auto offset = (_bbinfo.correct_transform_coords() ? T(1.f) : T(0.f));

    auto pred_ptr  = reinterpret_cast<T *>(_pred_boxes->buffer() + _pred_boxes->info()->offset_first_element_in_bytes());
    auto delta_ptr = reinterpret_cast<T *>(_deltas->buffer() + _deltas->info()->offset_first_element_in_bytes());

    Iterator box_it(_boxes, window);
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
            const T      dx       = delta_ptr[delta_id] / T(_bbinfo.weights()[0]);
            const T      dy       = delta_ptr[delta_id + 1] / T(_bbinfo.weights()[1]);
            T            dw       = delta_ptr[delta_id + 2] / T(_bbinfo.weights()[2]);
            T            dh       = delta_ptr[delta_id + 3] / T(_bbinfo.weights()[3]);
            // Clip dw and dh
            dw = std::min(dw, T(_bbinfo.bbox_xform_clip()));
            dh = std::min(dh, T(_bbinfo.bbox_xform_clip()));
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

void NEBoundingBoxTransformKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    switch(_boxes->info()->data_type())
    {
        case DataType::F32:
        {
            internal_run<float>(window);
            break;
        }
        case DataType::QASYMM16:
        {
            internal_run<uint16_t>(window);
            break;
        }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
        {
            internal_run<float16_t>(window);
            break;
        }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        default:
        {
            ARM_COMPUTE_ERROR("Data type not supported");
        }
    }
}
} // namespace arm_compute
