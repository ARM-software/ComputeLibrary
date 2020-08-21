/*
 * Copyright (c) 2019 Arm Limited.
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
#include "arm_compute/core/NEON/kernels/NEGenerateProposalsLayerKernel.h"

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
Status validate_arguments(const ITensorInfo *anchors, const ITensorInfo *all_anchors, const ComputeAnchorsInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(anchors, all_anchors);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(anchors);
    ARM_COMPUTE_RETURN_ERROR_ON(anchors->dimension(0) != info.values_per_roi());
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(anchors, DataType::QSYMM16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(anchors->num_dimensions() > 2);
    if(all_anchors->total_size() > 0)
    {
        const size_t feature_height = info.feat_height();
        const size_t feature_width  = info.feat_width();
        const size_t num_anchors    = anchors->dimension(1);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(all_anchors, anchors);
        ARM_COMPUTE_RETURN_ERROR_ON(all_anchors->num_dimensions() > 2);
        ARM_COMPUTE_RETURN_ERROR_ON(all_anchors->dimension(0) != info.values_per_roi());
        ARM_COMPUTE_RETURN_ERROR_ON(all_anchors->dimension(1) != feature_height * feature_width * num_anchors);

        if(is_data_type_quantized(anchors->data_type()))
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(anchors, all_anchors);
        }
    }
    return Status{};
}

} // namespace

NEComputeAllAnchorsKernel::NEComputeAllAnchorsKernel()
    : _anchors(nullptr), _all_anchors(nullptr), _anchors_info(0.f, 0.f, 0.f)
{
}

void NEComputeAllAnchorsKernel::configure(const ITensor *anchors, ITensor *all_anchors, const ComputeAnchorsInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(anchors, all_anchors);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(anchors->info(), all_anchors->info(), info));

    // Metadata
    const size_t   num_anchors = anchors->info()->dimension(1);
    const DataType data_type   = anchors->info()->data_type();
    const float    width       = info.feat_width();
    const float    height      = info.feat_height();

    // Initialize the output if empty
    const TensorShape output_shape(info.values_per_roi(), width * height * num_anchors);
    auto_init_if_empty(*all_anchors->info(), TensorInfo(output_shape, 1, data_type, anchors->info()->quantization_info()));

    // Set instance variables
    _anchors      = anchors;
    _all_anchors  = all_anchors;
    _anchors_info = info;

    Window win = calculate_max_window(*all_anchors->info(), Steps(info.values_per_roi()));

    INEKernel::configure(win);
}

Status NEComputeAllAnchorsKernel::validate(const ITensorInfo *anchors, const ITensorInfo *all_anchors, const ComputeAnchorsInfo &info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(anchors, all_anchors, info));
    return Status{};
}

template <>
void NEComputeAllAnchorsKernel::internal_run<int16_t>(const Window &window)
{
    Iterator all_anchors_it(_all_anchors, window);
    Iterator anchors_it(_all_anchors, window);

    const size_t num_anchors = _anchors->info()->dimension(1);
    const float  stride      = 1.f / _anchors_info.spatial_scale();
    const size_t feat_width  = _anchors_info.feat_width();

    const UniformQuantizationInfo qinfo = _anchors->info()->quantization_info().uniform();

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const size_t anchor_offset = id.y() % num_anchors;

        const auto out_anchor_ptr = reinterpret_cast<int16_t *>(all_anchors_it.ptr());
        const auto anchor_ptr     = reinterpret_cast<int16_t *>(_anchors->ptr_to_element(Coordinates(0, anchor_offset)));

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

template <typename T>
void NEComputeAllAnchorsKernel::internal_run(const Window &window)
{
    Iterator all_anchors_it(_all_anchors, window);
    Iterator anchors_it(_all_anchors, window);

    const size_t num_anchors = _anchors->info()->dimension(1);
    const T      stride      = 1.f / _anchors_info.spatial_scale();
    const size_t feat_width  = _anchors_info.feat_width();

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const size_t anchor_offset = id.y() % num_anchors;

        const auto out_anchor_ptr = reinterpret_cast<T *>(all_anchors_it.ptr());
        const auto anchor_ptr     = reinterpret_cast<T *>(_anchors->ptr_to_element(Coordinates(0, anchor_offset)));

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

void NEComputeAllAnchorsKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    switch(_anchors->info()->data_type())
    {
        case DataType::QSYMM16:
        {
            internal_run<int16_t>(window);
            break;
        }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
        {
            internal_run<float16_t>(window);
            break;
        }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F32:
        {
            internal_run<float>(window);
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Data type not supported");
        }
    }
}
} // namespace arm_compute
