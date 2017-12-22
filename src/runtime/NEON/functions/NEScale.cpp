/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "support/ToolchainSupport.h"

#include <cmath>
#include <cstddef>
#include <utility>

using namespace arm_compute;

namespace
{
void precompute_dx_dy_offsets(ITensor *dx, ITensor *dy, ITensor *offsets, float wr, float hr, size_t input_element_size, SamplingPolicy sampling_policy)
{
    ARM_COMPUTE_ERROR_ON(nullptr == offsets);
    ARM_COMPUTE_ERROR_ON(sampling_policy != SamplingPolicy::CENTER);
    ARM_COMPUTE_UNUSED(sampling_policy);

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
            const float in_x  = (id.x() + 0.5f) * wr - 0.5f;
            const float in_y  = (id.y() + 0.5f) * hr - 0.5f;
            const int   in_xi = std::floor(in_x);
            const int   in_yi = std::floor(in_y);

            *reinterpret_cast<int32_t *>(offsets_it.ptr()) = in_xi * input_element_size;
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
            const size_t in_xi = (id.x() + 0.5f) * wr;

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
      _border_handler()
{
}

void NEScale::configure(ITensor *input, ITensor *output, InterpolationPolicy policy, BorderMode border_mode, PixelValue constant_border_value, SamplingPolicy sampling_policy)
{
    ARM_COMPUTE_ERROR_ON(nullptr == input);
    ARM_COMPUTE_ERROR_ON(nullptr == output);

    for(size_t i = 2; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_ERROR_ON(input->info()->dimension(i) != output->info()->dimension(i));
    }

    // Get the tensor shape
    const TensorShape shape(output->info()->dimension(0), output->info()->dimension(1));

    // Compute the ratio between source width/height and destination width/height
    const auto wr = static_cast<float>(input->info()->dimension(0)) / static_cast<float>(output->info()->dimension(0));
    const auto hr = static_cast<float>(input->info()->dimension(1)) / static_cast<float>(output->info()->dimension(1));

    // Get the element size of the input image
    const size_t input_element_size = input->info()->element_size();

    // Area interpolation behaves as Nearest Neighbour in case of up-sampling
    if(policy == InterpolationPolicy::AREA && wr <= 1.f && hr <= 1.f)
    {
        policy = InterpolationPolicy::NEAREST_NEIGHBOR;
    }

    // Check if the border mode is UNDEFINED
    const bool border_undefined = border_mode == BorderMode::UNDEFINED;

    switch(policy)
    {
        case InterpolationPolicy::NEAREST_NEIGHBOR:
        {
            TensorInfo tensor_info_offsets(shape, Format::S32);
            _offsets.allocator()->init(tensor_info_offsets);

            _scale_kernel.configure(input, nullptr, nullptr, &_offsets, output, policy, border_undefined, sampling_policy);

            // Allocate once the configure methods have been called
            _offsets.allocator()->allocate();

            // Pre-compute offsets for nearest interpolation
            precompute_dx_dy_offsets(nullptr, nullptr, &_offsets, wr, hr, input_element_size, sampling_policy);
            break;
        }
        case InterpolationPolicy::BILINEAR:
        {
            TensorInfo tensor_info_offsets(shape, Format::S32);
            TensorInfo tensor_info_dxdy(shape, Format::F32);

            _offsets.allocator()->init(tensor_info_offsets);
            _dx.allocator()->init(tensor_info_dxdy);
            _dy.allocator()->init(tensor_info_dxdy);

            _scale_kernel.configure(input, &_dx, &_dy, &_offsets, output, policy, border_undefined, sampling_policy);

            // Allocate once the configure methods have been called
            _offsets.allocator()->allocate();
            _dx.allocator()->allocate();
            _dy.allocator()->allocate();

            // Pre-compute dx, dy and offsets for bilinear interpolation
            precompute_dx_dy_offsets(&_dx, &_dy, &_offsets, wr, hr, input_element_size, sampling_policy);
            break;
        }
        case InterpolationPolicy::AREA:
        {
            _scale_kernel.configure(input, nullptr, nullptr, nullptr, output, policy, border_undefined);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Unsupported interpolation mode");
    }

    _border_handler.configure(input, _scale_kernel.border_size(), border_mode, PixelValue(constant_border_value));
}

void NEScale::run()
{
    NEScheduler::get().schedule(&_border_handler, Window::DimZ);
    NEScheduler::get().schedule(&_scale_kernel, Window::DimY);
}
