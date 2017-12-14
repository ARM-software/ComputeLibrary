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
#include "arm_compute/core/NEON/kernels/NEDeconvolutionLayerUpsampleKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

using namespace arm_compute;

NEDeconvolutionLayerUpsampleKernel::NEDeconvolutionLayerUpsampleKernel()
    : _offsets(nullptr), _input(nullptr), _output(nullptr)
{
}

BorderSize NEDeconvolutionLayerUpsampleKernel::border_size() const
{
    return BorderSize(1);
}

void NEDeconvolutionLayerUpsampleKernel::configure(const ITensor *input, const ITensor *offsets, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON(output->info()->dimension(0) == 0);
    ARM_COMPUTE_ERROR_ON(output->info()->dimension(1) == 0);

    for(size_t i = 2; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_ERROR_ON(input->info()->dimension(i) != output->info()->dimension(i));
    }

    _input   = input;
    _output  = output;
    _offsets = offsets;

    constexpr unsigned int num_elems_processed_per_iteration = 16;
    const int              border_offset                     = border_size().left;

    // Configure kernel window
    Window win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowRectangle  input_access(input->info(), -border_offset, -border_offset, input->info()->dimension(0) + border_offset, input->info()->dimension(1) + border_offset);
    AccessWindowHorizontal offsets_access(offsets->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win, input_access, offsets_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));

    INEKernel::configure(win);
}

void NEDeconvolutionLayerUpsampleKernel::scale_nearest(const Window &window)
{
    const size_t input_stride = _input->info()->strides_in_bytes()[1];

    // Compute the ratio between source height and destination height
    const auto hr = static_cast<float>(_input->info()->dimension(1)) / static_cast<float>(_output->info()->dimension(1));

    // Don't increment in X and Y direction for the input tensor
    // A pointer to the start of this plane is needed as base for the precomputed offsets
    Window win_in(window);
    win_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimY, Window::Dimension(0, 0, 0));

    Window win_off;
    win_off.set(Window::DimX, window[Window::DimX]);
    win_off.set(Window::DimY, window[Window::DimY]);

    for(size_t d = Window::DimZ; d < _offsets->info()->num_dimensions(); ++d)
    {
        win_off.set(d, Window::Dimension(0, 0, 0));
    }

    Iterator in(_input, win_in);
    Iterator out(_output, window);
    Iterator offsets(_offsets, win_off);

    switch(_input->info()->data_type())
    {
        case DataType::F32:
        {
            float32x4x4_t tmp =
            {
                {
                    vdupq_n_f32(0),
                    vdupq_n_f32(0)
                }
            };
            execute_window_loop(window, [&](const Coordinates & id)
            {
                const auto offsets_ptr = reinterpret_cast<const int32_t *>(offsets.ptr());

                const size_t in_yi      = (id.y() + 0.5f) * hr;
                const size_t offset_row = in_yi * input_stride;

                tmp.val[0] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[0] + offset_row), tmp.val[0], 0);
                tmp.val[0] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[4] + offset_row), tmp.val[0], 1);
                tmp.val[0] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[8] + offset_row), tmp.val[0], 2);
                tmp.val[0] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[12] + offset_row), tmp.val[0], 3);

                tmp.val[1] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[1] + offset_row), tmp.val[1], 0);
                tmp.val[1] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[5] + offset_row), tmp.val[1], 1);
                tmp.val[1] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[9] + offset_row), tmp.val[1], 2);
                tmp.val[1] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[13] + offset_row), tmp.val[1], 3);

                tmp.val[2] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[2] + offset_row), tmp.val[2], 0);
                tmp.val[2] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[6] + offset_row), tmp.val[2], 1);
                tmp.val[2] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[10] + offset_row), tmp.val[2], 2);
                tmp.val[2] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[14] + offset_row), tmp.val[2], 3);

                tmp.val[3] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[3] + offset_row), tmp.val[3], 0);
                tmp.val[3] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[7] + offset_row), tmp.val[3], 1);
                tmp.val[3] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[11] + offset_row), tmp.val[3], 2);
                tmp.val[3] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[15] + offset_row), tmp.val[3], 3);

                vst4q_f32(reinterpret_cast<float *>(out.ptr()), tmp);
            },
            in, offsets, out);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not supported");
            break;
    }
}

void NEDeconvolutionLayerUpsampleKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    scale_nearest(window);
}
