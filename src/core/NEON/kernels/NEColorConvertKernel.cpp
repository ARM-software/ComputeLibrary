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
#include "arm_compute/core/NEON/kernels/NEColorConvertKernel.h"

#include "arm_compute/core/AccessWindowAutoPadding.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IMultiImage.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/MultiImageInfo.h"
#include "arm_compute/core/NEON/NEColorConvertHelper.inl"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

using namespace arm_compute;

NEColorConvertKernel::NEColorConvertKernel()
    : _input(nullptr), _output(nullptr), _num_elems_processed_per_iteration(0), _func(nullptr)
{
}

void NEColorConvertKernel::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    switch(input->info()->format())
    {
        case Format::RGBA8888:
        {
            switch(output->info()->format())
            {
                case Format::RGB888:
                    _func                              = colorconvert_rgbx_to_rgb;
                    _num_elems_processed_per_iteration = 16;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not supported");
                    break;
            }
            break;
        }
        case Format::UYVY422:
        {
            switch(output->info()->format())
            {
                case Format::RGB888:
                    _func                              = colorconvert_yuyv_to_rgb<false, false>;
                    _num_elems_processed_per_iteration = 32;
                    break;
                case Format::RGBA8888:
                    _func                              = colorconvert_yuyv_to_rgb<false, true>;
                    _num_elems_processed_per_iteration = 32;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not supported");
                    break;
            }
            break;
        }
        case Format::YUYV422:
        {
            switch(output->info()->format())
            {
                case Format::RGB888:
                    _func                              = colorconvert_yuyv_to_rgb<true, false>;
                    _num_elems_processed_per_iteration = 32;
                    break;
                case Format::RGBA8888:
                    _func                              = colorconvert_yuyv_to_rgb<true, true>;
                    _num_elems_processed_per_iteration = 32;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not supported");
                    break;
            }
            break;
        }
        case Format::RGB888:
        {
            switch(output->info()->format())
            {
                case Format::RGBA8888:
                    _func                              = colorconvert_rgb_to_rgbx;
                    _num_elems_processed_per_iteration = 16;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not supported");
                    break;
            }
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not supported");
            break;
    }

    _input  = input;
    _output = output;

    // Configure kernel window
    Window                  win = calculate_max_window(*input->info(), Steps(_num_elems_processed_per_iteration));
    AccessWindowAutoPadding output_access(output->info());

    update_window_and_padding(win,
                              AccessWindowAutoPadding(input->info()),
                              output_access);

    output_access.set_valid_region();

    INEKernel::configure(win);
}

void NEColorConvertKernel::configure(const IMultiImage *input, IImage *output)
{
    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(output);

    switch(input->info()->format())
    {
        case Format::NV12:
        {
            switch(output->info()->format())
            {
                case Format::RGB888:
                    _func                              = colorconvert_nv12_to_rgb<true, false>;
                    _num_elems_processed_per_iteration = 32;
                    break;
                case Format::RGBA8888:
                    _func                              = colorconvert_nv12_to_rgb<true, true>;
                    _num_elems_processed_per_iteration = 32;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not supported");
                    break;
            }
            break;
        }
        case Format::NV21:
        {
            switch(output->info()->format())
            {
                case Format::RGB888:
                    _func                              = colorconvert_nv12_to_rgb<false, false>;
                    _num_elems_processed_per_iteration = 32;
                    break;
                case Format::RGBA8888:
                    _func                              = colorconvert_nv12_to_rgb<false, true>;
                    _num_elems_processed_per_iteration = 32;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not supported");
                    break;
            }
            break;
        }
        case Format::IYUV:
        {
            switch(output->info()->format())
            {
                case Format::RGB888:
                    _func                              = colorconvert_iyuv_to_rgb<false>;
                    _num_elems_processed_per_iteration = 32;
                    break;
                case Format::RGBA8888:
                    _func                              = colorconvert_iyuv_to_rgb<true>;
                    _num_elems_processed_per_iteration = 32;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not supported");
                    break;
            }
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not supported");
            break;
    }

    _input  = input;
    _output = output;

    // Configure kernel window
    Window win = calculate_max_window(*output->info(), Steps(_num_elems_processed_per_iteration));
    win.set_dimension_step(Window::DimY, 2);

    AccessWindowAutoPadding output_access(output->info());

    unsigned int input_plane_count = 3;

    if(input->info()->format() == Format::NV12 || input->info()->format() == Format::NV21)
    {
        input_plane_count = 2;
    }

    update_window_and_padding(win,
                              AccessWindowAutoPadding(input->plane(0)->info()),
                              AccessWindowAutoPadding(input->plane(1)->info()),
                              AccessWindowAutoPadding(input_plane_count == 2 ? nullptr : input->plane(2)->info()),
                              output_access);

    output_access.set_valid_region();

    INEKernel::configure(win);
}

void NEColorConvertKernel::configure(const IImage *input, IMultiImage *output)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);

    switch(input->info()->format())
    {
        case Format::RGB888:
        {
            switch(output->info()->format())
            {
                case Format::NV12:
                    _func                              = colorconvert_rgb_to_nv12<false>;
                    _num_elems_processed_per_iteration = 16;
                    break;
                case Format::IYUV:
                    _func                              = colorconvert_rgb_to_iyuv<false>;
                    _num_elems_processed_per_iteration = 16;
                    break;
                case Format::YUV444:
                    _func                              = colorconvert_rgb_to_yuv4<false>;
                    _num_elems_processed_per_iteration = 16;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not supported");
                    break;
            }
            break;
        }
        case Format::RGBA8888:
        {
            switch(output->info()->format())
            {
                case Format::NV12:
                    _func                              = colorconvert_rgb_to_nv12<true>;
                    _num_elems_processed_per_iteration = 16;
                    break;
                case Format::IYUV:
                    _func                              = colorconvert_rgb_to_iyuv<true>;
                    _num_elems_processed_per_iteration = 16;
                    break;
                case Format::YUV444:
                    _func                              = colorconvert_rgb_to_yuv4<true>;
                    _num_elems_processed_per_iteration = 16;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not supported");
                    break;
            }
            break;
        }
        case Format::UYVY422:
        {
            switch(output->info()->format())
            {
                case Format::NV12:
                    _func                              = colorconvert_yuyv_to_nv12<false>;
                    _num_elems_processed_per_iteration = 32;
                    break;
                case Format::IYUV:
                    _func                              = colorconvert_yuyv_to_iyuv<false>;
                    _num_elems_processed_per_iteration = 32;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not supported");
                    break;
            }
            break;
        }
        case Format::YUYV422:
        {
            switch(output->info()->format())
            {
                case Format::NV12:
                    _func                              = colorconvert_yuyv_to_nv12<true>;
                    _num_elems_processed_per_iteration = 32;
                    break;
                case Format::IYUV:
                    _func                              = colorconvert_yuyv_to_iyuv<true>;
                    _num_elems_processed_per_iteration = 32;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not supported");
                    break;
            }
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not supported");
            break;
    }

    _input  = input;
    _output = output;

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps(_num_elems_processed_per_iteration));
    if((input->info()->format() != Format::RGB888 || output->info()->format() != Format::YUV444) && (input->info()->format() != Format::RGBA8888 || output->info()->format() != Format::YUV444))
    {
        win.set_dimension_step(Window::DimY, 2);
    }

    unsigned int output_plane_count = 3;

    if(output->info()->format() == Format::NV12 || output->info()->format() == Format::NV21)
    {
        output_plane_count = 2;
    }

    AccessWindowAutoPadding output0_access(output->plane(0)->info());
    AccessWindowAutoPadding output1_access(output->plane(1)->info());
    AccessWindowAutoPadding output2_access(output_plane_count == 2 ? nullptr : output->plane(2)->info());

    update_window_and_padding(win,
                              AccessWindowAutoPadding(input->info()),
                              output0_access,
                              output1_access,
                              output2_access);

    output0_access.set_valid_region();
    output1_access.set_valid_region();
    output2_access.set_valid_region();

    INEKernel::configure(win);
}

void NEColorConvertKernel::configure(const IMultiImage *input, IMultiImage *output)
{
    switch(input->info()->format())
    {
        case Format::NV12:
        {
            switch(output->info()->format())
            {
                case Format::IYUV:
                    _func                              = colorconvert_nv12_to_iyuv<true>;
                    _num_elems_processed_per_iteration = 32;
                    break;
                case Format::YUV444:
                    _func                              = colorconvert_nv12_to_yuv4<true>;
                    _num_elems_processed_per_iteration = 32;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not supported");
                    break;
            }
            break;
        }
        case Format::NV21:
        {
            switch(output->info()->format())
            {
                case Format::IYUV:
                    _func                              = colorconvert_nv12_to_iyuv<false>;
                    _num_elems_processed_per_iteration = 32;
                    break;
                case Format::YUV444:
                    _func                              = colorconvert_nv12_to_yuv4<false>;
                    _num_elems_processed_per_iteration = 32;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not supported");
                    break;
            }
            break;
        }
        case Format::IYUV:
        {
            switch(output->info()->format())
            {
                case Format::NV12:
                    _func                              = colorconvert_iyuv_to_nv12;
                    _num_elems_processed_per_iteration = 32;
                    break;
                case Format::YUV444:
                    _func                              = colorconvert_iyuv_to_yuv4;
                    _num_elems_processed_per_iteration = 32;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not supported");
                    break;
            }
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not supported");
            break;
    }

    _input  = input;
    _output = output;

    // Configure kernel window
    Window win = calculate_max_window(*input->plane(0)->info(), Steps(_num_elems_processed_per_iteration));
    win.set_dimension_step(Window::DimY, 2);

    unsigned int input_plane_count = 3;

    if(input->info()->format() == Format::NV12 || input->info()->format() == Format::NV21)
    {
        input_plane_count = 2;
    }

    unsigned int output_plane_count = 3;

    if(output->info()->format() == Format::NV12 || output->info()->format() == Format::NV21)
    {
        output_plane_count = 2;
    }

    AccessWindowAutoPadding output0_access(output->plane(0)->info());
    AccessWindowAutoPadding output1_access(output->plane(1)->info());
    AccessWindowAutoPadding output2_access(output_plane_count == 2 ? nullptr : output->plane(2)->info());

    update_window_and_padding(win,
                              AccessWindowAutoPadding(input->plane(0)->info()),
                              AccessWindowAutoPadding(input->plane(1)->info()),
                              AccessWindowAutoPadding(input_plane_count == 2 ? nullptr : input->plane(2)->info()),
                              output0_access,
                              output1_access,
                              output2_access);

    output0_access.set_valid_region();
    output1_access.set_valid_region();
    output2_access.set_valid_region();

    INEKernel::configure(win);
}

void NEColorConvertKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (*_func)(_input, _output, window);
}
