/*
 * Copyright (c) 2016-2020 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEScale.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/Rounding.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "src/core/utils/ScaleUtils.h"

#include <cmath>
#include <cstddef>
#include <utility>

namespace arm_compute
{
namespace
{
void precompute_dx_dy_offsets(ITensor *dx, ITensor *dy, ITensor *offsets, float wr, float hr, size_t input_element_size, SamplingPolicy sampling_policy, bool align_corners)
{
    ARM_COMPUTE_ERROR_ON(nullptr == offsets);
    ARM_COMPUTE_UNUSED(sampling_policy);
    float sampling_offset = 0.0f;
    if(sampling_policy == SamplingPolicy::CENTER)
    {
        sampling_offset = 0.5f;
    }

    Window win;
    win.set(Window::DimX, Window::Dimension(0, offsets->info()->dimension(0), 1));
    win.set(Window::DimY, Window::Dimension(0, offsets->info()->dimension(1), 1));

    if(dx != nullptr && dy != nullptr)
    {
        // Pre-compute the offset and pixel's distance for BILINEAR interpolation
        Iterator offsets_it(offsets, win);
        Iterator dx_it(dx, win);
        Iterator dy_it(dy, win);

        execute_window_loop(win, [&](const Coordinates & id)
        {
            const float in_x  = (id.x() + sampling_offset) * wr - sampling_offset;
            const float in_y  = (id.y() + sampling_offset) * hr - sampling_offset;
            const int   in_xi = std::floor(in_x);
            const int   in_yi = std::floor(in_y);

            *reinterpret_cast<int32_t *>(offsets_it.ptr()) = in_xi * static_cast<int>(input_element_size);
            *reinterpret_cast<float *>(dx_it.ptr())        = in_x - in_xi;
            *reinterpret_cast<float *>(dy_it.ptr())        = in_y - in_yi;
        },
        offsets_it, dx_it, dy_it);
    }
    else
    {
        // Pre-compute the offset for NEAREST interpolation
        Iterator offsets_it(offsets, win);

        execute_window_loop(win, [&](const Coordinates & id)
        {
            const float float_in_xi = (id.x() + sampling_offset) * wr;
            const auto  in_xi       = static_cast<size_t>(align_corners ? arm_compute::utils::rounding::round_half_away_from_zero(float_in_xi) : std::floor(float_in_xi));

            *reinterpret_cast<int32_t *>(offsets_it.ptr()) = in_xi * input_element_size;
        },
        offsets_it);
    }
}
} // namespace

NEScale::NEScale() // NOLINT
    : _offsets(),
      _dx(),
      _dy(),
      _scale_kernel(),
      _border_handler(),
      _use_padding(true)
{
}

void NEScale::configure(ITensor *input, ITensor *output, const ScaleKernelInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(NEScale::validate(input->info(), output->info(), info));

    _use_padding                     = info.use_padding;
    const bool is_align_corners_used = info.align_corners && arm_compute::scale_utils::is_align_corners_allowed_sampling_policy(info.sampling_policy)
                                       && arm_compute::scale_utils::is_align_corners_allowed_output_shape(output->info()->tensor_shape(), output->info()->data_layout());

    // Get data layout and width/height indices
    const DataLayout data_layout = input->info()->data_layout();
    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    // Get the tensor shape
    const TensorShape shape(output->info()->dimension(idx_width), output->info()->dimension(idx_height));

    // Compute the ratio between source width/height and destination width/height
    const auto wr = arm_compute::scale_utils::calculate_resize_ratio(input->info()->dimension(idx_width), output->info()->dimension(idx_width), is_align_corners_used);
    const auto hr = arm_compute::scale_utils::calculate_resize_ratio(input->info()->dimension(idx_height), output->info()->dimension(idx_height), is_align_corners_used);

    // Get the element size of the input image
    const size_t input_element_size = input->info()->element_size();

    // Area interpolation behaves as Nearest Neighbour in case of up-sampling
    const auto policy_to_use = (info.interpolation_policy == InterpolationPolicy::AREA && wr <= 1.f && hr <= 1.f) ? InterpolationPolicy::NEAREST_NEIGHBOR : info.interpolation_policy;

    switch(policy_to_use)
    {
        case InterpolationPolicy::NEAREST_NEIGHBOR:
        {
            TensorInfo tensor_info_offsets(shape, Format::S32);
            _offsets.allocator()->init(tensor_info_offsets);

            _scale_kernel.configure(input, nullptr, nullptr, &_offsets, output, info);

            // Allocate once the configure methods have been called
            _offsets.allocator()->allocate();

            // Pre-compute offsets for nearest interpolation
            precompute_dx_dy_offsets(nullptr, nullptr, &_offsets, wr, hr, input_element_size, info.sampling_policy, is_align_corners_used);
            break;
        }
        case InterpolationPolicy::BILINEAR:
        {
            TensorInfo tensor_info_offsets(shape, Format::S32);
            TensorInfo tensor_info_dxdy(shape, Format::F32);

            _offsets.allocator()->init(tensor_info_offsets);
            _dx.allocator()->init(tensor_info_dxdy);
            _dy.allocator()->init(tensor_info_dxdy);

            _scale_kernel.configure(input, &_dx, &_dy, &_offsets, output, info);

            // Allocate once the configure methods have been called
            _offsets.allocator()->allocate();
            _dx.allocator()->allocate();
            _dy.allocator()->allocate();

            // Pre-compute dx, dy and offsets for bilinear interpolation
            precompute_dx_dy_offsets(&_dx, &_dy, &_offsets, wr, hr, input_element_size, info.sampling_policy, is_align_corners_used);
            break;
        }
        case InterpolationPolicy::AREA:
        {
            _scale_kernel.configure(input, nullptr, nullptr, nullptr, output, info);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Unsupported interpolation mode");
    }
    if(info.use_padding)
    {
        _border_handler.configure(input, _scale_kernel.border_size(), info.border_mode, info.constant_border_value);
    }
}

void NEScale::configure(ITensor *input, ITensor *output, InterpolationPolicy policy, BorderMode border_mode, PixelValue constant_border_value, SamplingPolicy sampling_policy, bool use_padding,
                        bool align_corners)
{
    configure(input, output, ScaleKernelInfo{ policy, border_mode, constant_border_value, sampling_policy, use_padding, align_corners });
}

Status NEScale::validate(const ITensorInfo *input, const ITensorInfo *output, const ScaleKernelInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(info.sampling_policy != SamplingPolicy::CENTER && info.sampling_policy != SamplingPolicy::TOP_LEFT);

    ITensorInfo *offsets = nullptr;
    ITensorInfo *dx      = nullptr;
    ITensorInfo *dy      = nullptr;

    // Get data layout and width/height indices
    const DataLayout data_layout = input->data_layout();
    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    // Get the tensor shape of auxilary buffers
    const TensorShape shape(output->dimension(idx_width), output->dimension(idx_height));

    TensorInfo tensor_info_offsets(shape, Format::S32);
    TensorInfo tensor_info_dx(shape, Format::F32);
    TensorInfo tensor_info_dy(shape, Format::F32);

    switch(info.interpolation_policy)
    {
        case InterpolationPolicy::NEAREST_NEIGHBOR:
            offsets = &tensor_info_offsets;
            break;
        case InterpolationPolicy::BILINEAR:
            offsets = &tensor_info_offsets;
            dx      = &tensor_info_dx;
            dy      = &tensor_info_dy;
            break;
        default:
            break;
    }

    ARM_COMPUTE_RETURN_ON_ERROR(NEScaleKernel::validate(input->clone().get(), dx, dy, offsets, output->clone().get(), info));
    return Status{};
}

Status NEScale::validate(const ITensorInfo *input, const ITensorInfo *output, InterpolationPolicy policy,
                         BorderMode border_mode, PixelValue constant_border_value, SamplingPolicy sampling_policy, bool use_padding, bool align_corners)
{
    ARM_COMPUTE_RETURN_ON_ERROR(NEScale::validate(input, output, ScaleKernelInfo{ policy, border_mode, constant_border_value, sampling_policy, use_padding, align_corners }));
    return Status{};
}

void NEScale::run()
{
    if(_use_padding)
    {
        NEScheduler::get().schedule(&_border_handler, Window::DimZ);
    }
    NEScheduler::get().schedule(&_scale_kernel, Window::DimY);
}
} // namespace arm_compute
