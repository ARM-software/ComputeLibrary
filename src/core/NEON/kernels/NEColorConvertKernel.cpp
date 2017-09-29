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

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
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
    : _input(nullptr), _output(nullptr), _func(nullptr)
{
}

void NEColorConvertKernel::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    set_shape_if_empty(*output->info(), input->info()->tensor_shape());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, output);

    unsigned int num_elems_processed_per_iteration = 0;

    switch(input->info()->format())
    {
        case Format::RGBA8888:
        {
            switch(output->info()->format())
            {
                case Format::RGB888:
                    _func                             = colorconvert_rgbx_to_rgb;
                    num_elems_processed_per_iteration = 16;
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
                    _func                             = colorconvert_yuyv_to_rgb<false, false>;
                    num_elems_processed_per_iteration = 32;
                    break;
                case Format::RGBA8888:
                    _func                             = colorconvert_yuyv_to_rgb<false, true>;
                    num_elems_processed_per_iteration = 32;
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
                    _func                             = colorconvert_yuyv_to_rgb<true, false>;
                    num_elems_processed_per_iteration = 32;
                    break;
                case Format::RGBA8888:
                    _func                             = colorconvert_yuyv_to_rgb<true, true>;
                    num_elems_processed_per_iteration = 32;
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
                    _func                             = colorconvert_rgb_to_rgbx;
                    num_elems_processed_per_iteration = 16;
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
    Window                 win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal input_access(input->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, input->info()->valid_region());

    INEKernel::configure(win);
}

void NEColorConvertKernel::configure(const IMultiImage *input, IImage *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(output);

    set_shape_if_empty(*output->info(), input->plane(0)->info()->tensor_shape());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input->plane(0), output);

    unsigned int num_elems_processed_per_iteration = 0;

    switch(input->info()->format())
    {
        case Format::NV12:
        {
            switch(output->info()->format())
            {
                case Format::RGB888:
                    _func                             = colorconvert_nv12_to_rgb<true, false>;
                    num_elems_processed_per_iteration = 32;
                    break;
                case Format::RGBA8888:
                    _func                             = colorconvert_nv12_to_rgb<true, true>;
                    num_elems_processed_per_iteration = 32;
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
                    _func                             = colorconvert_nv12_to_rgb<false, false>;
                    num_elems_processed_per_iteration = 32;
                    break;
                case Format::RGBA8888:
                    _func                             = colorconvert_nv12_to_rgb<false, true>;
                    num_elems_processed_per_iteration = 32;
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
                    _func                             = colorconvert_iyuv_to_rgb<false>;
                    num_elems_processed_per_iteration = 32;
                    break;
                case Format::RGBA8888:
                    _func                             = colorconvert_iyuv_to_rgb<true>;
                    num_elems_processed_per_iteration = 32;
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
    Window win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration));
    win.set_dimension_step(Window::DimY, 2);

    unsigned int input_plane_count = 3;

    if(input->info()->format() == Format::NV12 || input->info()->format() == Format::NV21)
    {
        input_plane_count = 2;
    }

    AccessWindowHorizontal input0_access(input->plane(0)->info(), 0, num_elems_processed_per_iteration);
    AccessWindowRectangle  input1_access(input->plane(1)->info(), 0, 0, num_elems_processed_per_iteration, 1, 0.5f, 0.5f);
    AccessWindowRectangle  input2_access(input_plane_count == 2 ? nullptr : input->plane(2)->info(), 0, 0, num_elems_processed_per_iteration, 1, 0.5f, 0.5f);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win,
                              input0_access, input1_access, input2_access,
                              output_access);

    ValidRegion intersect_region = intersect_valid_regions(input->plane(0)->info()->valid_region(),
                                                           input->plane(1)->info()->valid_region());

    if(input_plane_count == 3)
    {
        intersect_region = intersect_valid_regions(intersect_region, input->plane(2)->info()->valid_region());
    }

    output_access.set_valid_region(win, intersect_region);

    INEKernel::configure(win);
}

void NEColorConvertKernel::configure(const IImage *input, IMultiImage *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);

    set_shape_if_empty(*output->plane(0)->info(), input->info()->tensor_shape());

    switch(output->info()->format())
    {
        case Format::NV12:
        {
            TensorShape subsampled_shape = input->info()->tensor_shape();
            subsampled_shape.set(0, subsampled_shape[0] / 2);
            subsampled_shape.set(1, subsampled_shape[1] / 2);

            set_shape_if_empty(*output->plane(1)->info(), subsampled_shape);

            ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(subsampled_shape, output->plane(1)->info()->tensor_shape());
            break;
        }
        case Format::IYUV:
        {
            TensorShape subsampled_shape = input->info()->tensor_shape();
            subsampled_shape.set(0, subsampled_shape[0] / 2);
            subsampled_shape.set(1, subsampled_shape[1] / 2);

            set_shape_if_empty(*output->plane(1)->info(), subsampled_shape);
            set_shape_if_empty(*output->plane(2)->info(), subsampled_shape);

            ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(subsampled_shape, output->plane(1)->info()->tensor_shape());
            ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(subsampled_shape, output->plane(2)->info()->tensor_shape());
            break;
        }
        case Format::YUV444:
            set_shape_if_empty(*output->plane(1)->info(), input->info()->tensor_shape());
            set_shape_if_empty(*output->plane(2)->info(), input->info()->tensor_shape());

            ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, output->plane(1));
            ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, output->plane(2));
            break;
        default:
            ARM_COMPUTE_ERROR("Not supported");
    }

    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, output->plane(0));

    unsigned int num_elems_processed_per_iteration = 0;

    switch(input->info()->format())
    {
        case Format::RGB888:
        {
            switch(output->info()->format())
            {
                case Format::NV12:
                    _func                             = colorconvert_rgb_to_nv12<false>;
                    num_elems_processed_per_iteration = 16;
                    break;
                case Format::IYUV:
                    _func                             = colorconvert_rgb_to_iyuv<false>;
                    num_elems_processed_per_iteration = 16;
                    break;
                case Format::YUV444:
                    _func                             = colorconvert_rgb_to_yuv4<false>;
                    num_elems_processed_per_iteration = 16;
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
                    _func                             = colorconvert_rgb_to_nv12<true>;
                    num_elems_processed_per_iteration = 16;
                    break;
                case Format::IYUV:
                    _func                             = colorconvert_rgb_to_iyuv<true>;
                    num_elems_processed_per_iteration = 16;
                    break;
                case Format::YUV444:
                    _func                             = colorconvert_rgb_to_yuv4<true>;
                    num_elems_processed_per_iteration = 16;
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
                    _func                             = colorconvert_yuyv_to_nv12<false>;
                    num_elems_processed_per_iteration = 32;
                    break;
                case Format::IYUV:
                    _func                             = colorconvert_yuyv_to_iyuv<false>;
                    num_elems_processed_per_iteration = 32;
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
                    _func                             = colorconvert_yuyv_to_nv12<true>;
                    num_elems_processed_per_iteration = 32;
                    break;
                case Format::IYUV:
                    _func                             = colorconvert_yuyv_to_iyuv<true>;
                    num_elems_processed_per_iteration = 32;
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
    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    float sub_sampling = 1.f;

    if((input->info()->format() != Format::RGB888 || output->info()->format() != Format::YUV444) && (input->info()->format() != Format::RGBA8888 || output->info()->format() != Format::YUV444))
    {
        win.set_dimension_step(Window::DimY, 2);
        sub_sampling = 0.5f;
    }

    unsigned int output_plane_count = 3;

    if(output->info()->format() == Format::NV12 || output->info()->format() == Format::NV21)
    {
        output_plane_count = 2;
    }

    AccessWindowHorizontal output0_access(output->plane(0)->info(), 0, num_elems_processed_per_iteration);
    AccessWindowRectangle  output1_access(output->plane(1)->info(), 0, 0, num_elems_processed_per_iteration, 1, sub_sampling, sub_sampling);
    AccessWindowRectangle  output2_access(output_plane_count == 2 ? nullptr : output->plane(2)->info(), 0, 0, num_elems_processed_per_iteration, 1, sub_sampling, sub_sampling);

    update_window_and_padding(win,
                              AccessWindowHorizontal(input->info(), 0, num_elems_processed_per_iteration),
                              output0_access,
                              output1_access,
                              output2_access);

    output0_access.set_valid_region(win, input->info()->valid_region());
    output1_access.set_valid_region(win, input->info()->valid_region());
    output2_access.set_valid_region(win, input->info()->valid_region());

    INEKernel::configure(win);
}

void NEColorConvertKernel::configure(const IMultiImage *input, IMultiImage *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_ON(input == output);

    set_shape_if_empty(*output->plane(0)->info(), input->plane(0)->info()->tensor_shape());

    switch(output->info()->format())
    {
        case Format::NV12:
        {
            TensorShape subsampled_shape = input->plane(0)->info()->tensor_shape();
            subsampled_shape.set(0, subsampled_shape[0] / 2);
            subsampled_shape.set(1, subsampled_shape[1] / 2);

            set_shape_if_empty(*output->plane(1)->info(), subsampled_shape);

            ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(subsampled_shape, output->plane(1)->info()->tensor_shape());
            break;
        }
        case Format::IYUV:
        {
            TensorShape subsampled_shape = input->plane(0)->info()->tensor_shape();
            subsampled_shape.set(0, subsampled_shape[0] / 2);
            subsampled_shape.set(1, subsampled_shape[1] / 2);

            set_shape_if_empty(*output->plane(1)->info(), subsampled_shape);
            set_shape_if_empty(*output->plane(2)->info(), subsampled_shape);

            ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(subsampled_shape, output->plane(1)->info()->tensor_shape());
            ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(subsampled_shape, output->plane(2)->info()->tensor_shape());
            break;
        }
        case Format::YUV444:
            set_shape_if_empty(*output->plane(1)->info(), input->plane(0)->info()->tensor_shape());
            set_shape_if_empty(*output->plane(2)->info(), input->plane(0)->info()->tensor_shape());

            ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input->plane(0), output->plane(1));
            ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input->plane(0), output->plane(2));
            break;
        default:
            ARM_COMPUTE_ERROR("Not supported");
    }

    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input->plane(0), output->plane(0));

    switch(input->info()->format())
    {
        case Format::NV12:
        {
            switch(output->info()->format())
            {
                case Format::IYUV:
                    _func = colorconvert_nv12_to_iyuv<true>;
                    break;
                case Format::YUV444:
                    _func = colorconvert_nv12_to_yuv4<true>;
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
                    _func = colorconvert_nv12_to_iyuv<false>;
                    break;
                case Format::YUV444:
                    _func = colorconvert_nv12_to_yuv4<false>;
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
                    _func = colorconvert_iyuv_to_nv12;
                    break;
                case Format::YUV444:
                    _func = colorconvert_iyuv_to_yuv4;
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

    constexpr unsigned int num_elems_processed_per_iteration = 32;
    constexpr float        input_sub_sampling                = 0.5f;
    const float            output_sub_sampling               = output->info()->format() == Format::YUV444 ? 1.f : 0.5f;

    // Configure kernel window
    Window win = calculate_max_window(*input->plane(0)->info(), Steps(num_elems_processed_per_iteration));
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

    AccessWindowHorizontal output0_access(output->plane(0)->info(), 0, num_elems_processed_per_iteration);
    AccessWindowRectangle  output1_access(output->plane(1)->info(), 0, 0, num_elems_processed_per_iteration, 1, output_sub_sampling, output_sub_sampling);
    AccessWindowRectangle  output2_access(output_plane_count == 2 ? nullptr : output->plane(2)->info(), 0, 0, num_elems_processed_per_iteration, 1, output_sub_sampling, output_sub_sampling);

    update_window_and_padding(win,
                              AccessWindowHorizontal(input->plane(0)->info(), 0, num_elems_processed_per_iteration),
                              AccessWindowRectangle(input->plane(1)->info(), 0, 0, num_elems_processed_per_iteration, 1, input_sub_sampling, input_sub_sampling),
                              AccessWindowRectangle(input_plane_count == 2 ? nullptr : input->plane(2)->info(), 0, 0, num_elems_processed_per_iteration, 1, input_sub_sampling, input_sub_sampling),
                              output0_access,
                              output1_access,
                              output2_access);

    ValidRegion intersect_region = intersect_valid_regions(input->plane(0)->info()->valid_region(),
                                                           input->plane(1)->info()->valid_region());

    if(input_plane_count == 3)
    {
        intersect_region = intersect_valid_regions(intersect_region, input->plane(2)->info()->valid_region());
    }

    output0_access.set_valid_region(win, intersect_region);
    output1_access.set_valid_region(win, intersect_region);
    output2_access.set_valid_region(win, intersect_region);

    INEKernel::configure(win);
}

void NEColorConvertKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (*_func)(_input, _output, window);
}
