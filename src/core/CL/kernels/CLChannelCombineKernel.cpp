/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLChannelCombineKernel.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLMultiImage.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/MultiImageInfo.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <set>
#include <string>

namespace arm_compute
{
namespace
{
constexpr unsigned int num_elems_processed_per_iteration = 16;
} // namespace

CLChannelCombineKernel::CLChannelCombineKernel()
    : _planes{ { nullptr } }, _output(nullptr), _output_multi(nullptr), _x_subsampling{ { 1, 1, 1 } }, _y_subsampling{ { 1, 1, 1 } }
{
}

void CLChannelCombineKernel::configure(const ICLTensor *plane0, const ICLTensor *plane1, const ICLTensor *plane2, const ICLTensor *plane3, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(plane0, plane1, plane2, output);
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(plane0);
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(plane1);
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(plane2);
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(output);

    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(plane0, Format::U8);
    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(plane1, Format::U8);
    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(plane2, Format::U8);
    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(output, Format::RGB888, Format::RGBA8888, Format::YUYV422, Format::UYVY422);

    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(plane0, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(plane1, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(plane2, 1, DataType::U8);

    const Format output_format = output->info()->format();

    // Check if horizontal dimension of Y plane is even and validate horizontal sub-sampling dimensions for U and V planes
    if(Format::YUYV422 == output_format || Format::UYVY422 == output_format)
    {
        // Validate Y plane of input and output
        ARM_COMPUTE_ERROR_ON_TENSORS_NOT_EVEN(output_format, plane0, output);

        // Validate U and V plane of the input
        ARM_COMPUTE_ERROR_ON_TENSORS_NOT_SUBSAMPLED(output_format, plane0->info()->tensor_shape(), plane1, plane2);
    }

    _planes[0] = plane0;
    _planes[1] = plane1;
    _planes[2] = plane2;
    _planes[3] = nullptr;

    // Validate the last input tensor only for RGBA format
    if(Format::RGBA8888 == output_format)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(plane3);
        ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(plane3);

        ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(plane3, Format::U8);
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(plane3, 1, DataType::U8);

        _planes[3] = plane3;
    }

    _output       = output;
    _output_multi = nullptr;

    // Half the processed elements for U and V channels due to horizontal sub-sampling of 2
    if(Format::YUYV422 == output_format || Format::UYVY422 == output_format)
    {
        _x_subsampling[1] = 2;
        _x_subsampling[2] = 2;
    }

    // Create kernel
    std::string kernel_name = "channel_combine_" + string_from_format(output_format);
    _kernel                 = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name));

    // Configure window
    Window win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal plane0_access(plane0->info(), 0, num_elems_processed_per_iteration);
    AccessWindowRectangle  plane1_access(plane1->info(), 0, 0, num_elems_processed_per_iteration, 1, 1.f / _x_subsampling[1], 1.f / _y_subsampling[1]);
    AccessWindowRectangle  plane2_access(plane2->info(), 0, 0, num_elems_processed_per_iteration, 1, 1.f / _x_subsampling[2], 1.f / _y_subsampling[2]);
    AccessWindowHorizontal plane3_access(plane3 == nullptr ? nullptr : plane3->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win, plane0_access, plane1_access, plane2_access, plane3_access, output_access);

    ValidRegion valid_region = intersect_valid_regions(plane0->info()->valid_region(),
                                                       plane1->info()->valid_region(),
                                                       plane2->info()->valid_region());
    if(plane3 != nullptr)
    {
        valid_region = intersect_valid_regions(plane3->info()->valid_region(), valid_region);
    }
    output_access.set_valid_region(win, ValidRegion(valid_region.anchor, output->info()->tensor_shape()));

    ICLKernel::configure_internal(win);
}

void CLChannelCombineKernel::configure(const ICLImage *plane0, const ICLImage *plane1, const ICLImage *plane2, ICLMultiImage *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(plane0, plane1, plane2, output);
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(plane0);
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(plane1);
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(plane2);

    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(plane0, Format::U8);
    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(plane1, Format::U8);
    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(plane2, Format::U8);
    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(output, Format::NV12, Format::NV21, Format::IYUV, Format::YUV444);

    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(plane0, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(plane1, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(plane2, 1, DataType::U8);

    const Format output_format = output->info()->format();

    // Validate shape of Y plane to be even and shape of sub-sampling dimensions for U and V planes
    // Perform validation only for formats which require sub-sampling.
    if(Format::YUV444 != output_format)
    {
        // Validate Y plane of input and output
        ARM_COMPUTE_ERROR_ON_TENSORS_NOT_EVEN(output_format, plane0, output->plane(0));

        // Validate U and V plane of the input
        ARM_COMPUTE_ERROR_ON_TENSORS_NOT_SUBSAMPLED(output_format, plane0->info()->tensor_shape(), plane1, plane2);

        // Validate second plane U (NV12 and NV21 have a UV88 combined plane while IYUV has only the U plane)
        // MultiImage generates the correct tensor shape but also check in case the tensor shape of planes was changed to a wrong size
        ARM_COMPUTE_ERROR_ON_TENSORS_NOT_SUBSAMPLED(output_format, plane0->info()->tensor_shape(), output->plane(1));

        // Validate the last plane V of format IYUV
        if(Format::IYUV == output_format)
        {
            // Validate Y plane of the output
            ARM_COMPUTE_ERROR_ON_TENSORS_NOT_SUBSAMPLED(output_format, plane0->info()->tensor_shape(), output->plane(2));
        }
    }

    // Set input tensors
    _planes[0] = plane0;
    _planes[1] = plane1;
    _planes[2] = plane2;
    _planes[3] = nullptr;

    // Set output tensor
    _output       = nullptr;
    _output_multi = output;

    bool has_two_planars = false;

    // Set sub-sampling parameters for each plane
    std::string           kernel_name;
    std::set<std::string> build_opts;

    if(Format::NV12 == output_format || Format::NV21 == output_format)
    {
        _x_subsampling = { { 1, 2, 2 } };
        _y_subsampling = { { 1, 2, 2 } };
        kernel_name    = "channel_combine_NV";
        build_opts.emplace(Format::NV12 == output_format ? "-DNV12" : "-DNV21");
        has_two_planars = true;
    }
    else
    {
        if(Format::IYUV == output_format)
        {
            _x_subsampling = { { 1, 2, 2 } };
            _y_subsampling = { { 1, 2, 2 } };
        }

        kernel_name = "copy_planes_3p";
        build_opts.emplace(Format::IYUV == output_format ? "-DIYUV" : "-DYUV444");
    }

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts));

    // Configure window
    Window win = calculate_max_window(*plane0->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowRectangle input_plane0_access(plane0->info(), 0, 0, num_elems_processed_per_iteration, 1.f);
    AccessWindowRectangle input_plane1_access(plane1->info(), 0, 0, num_elems_processed_per_iteration, 1.f, 1.f / _x_subsampling[1], 1.f / _y_subsampling[1]);
    AccessWindowRectangle input_plane2_access(plane2->info(), 0, 0, num_elems_processed_per_iteration, 1.f, 1.f / _x_subsampling[2], 1.f / _y_subsampling[2]);
    AccessWindowRectangle output_plane0_access(output->plane(0)->info(), 0, 0, num_elems_processed_per_iteration, 1.f, 1.f, 1.f / _y_subsampling[1]);
    AccessWindowRectangle output_plane1_access(output->plane(1)->info(), 0, 0, num_elems_processed_per_iteration, 1.f, 1.f / _x_subsampling[1], 1.f / _y_subsampling[1]);
    AccessWindowRectangle output_plane2_access(has_two_planars ? nullptr : output->plane(2)->info(), 0, 0, num_elems_processed_per_iteration, 1.f, 1.f / _x_subsampling[2], 1.f / _y_subsampling[2]);

    update_window_and_padding(win,
                              input_plane0_access, input_plane1_access, input_plane2_access,
                              output_plane0_access, output_plane1_access, output_plane2_access);

    ValidRegion plane0_valid_region  = plane0->info()->valid_region();
    ValidRegion output_plane1_region = has_two_planars ? intersect_valid_regions(plane1->info()->valid_region(), plane2->info()->valid_region()) : plane2->info()->valid_region();
    output_plane0_access.set_valid_region(win, ValidRegion(plane0_valid_region.anchor, output->plane(0)->info()->tensor_shape()));
    output_plane1_access.set_valid_region(win, ValidRegion(output_plane1_region.anchor, output->plane(1)->info()->tensor_shape()));
    output_plane2_access.set_valid_region(win, ValidRegion(plane2->info()->valid_region().anchor, output->plane(2)->info()->tensor_shape()));

    ICLKernel::configure_internal(win);
}

void CLChannelCombineKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice = window.first_slice_window_2D();
    slice.set_dimension_step(Window::DimY, 1);

    do
    {
        // Subsampling in plane 1
        Window win_sub_plane1(slice);
        win_sub_plane1.set(Window::DimX, Window::Dimension(win_sub_plane1.x().start() / _x_subsampling[1], win_sub_plane1.x().end() / _x_subsampling[1], win_sub_plane1.x().step() / _x_subsampling[1]));
        win_sub_plane1.set(Window::DimY, Window::Dimension(win_sub_plane1.y().start() / _y_subsampling[1], win_sub_plane1.y().end() / _y_subsampling[1], 1));

        // Subsampling in plane 2
        Window win_sub_plane2(slice);
        win_sub_plane2.set(Window::DimX, Window::Dimension(win_sub_plane2.x().start() / _x_subsampling[2], win_sub_plane2.x().end() / _x_subsampling[2], win_sub_plane2.x().step() / _x_subsampling[2]));
        win_sub_plane2.set(Window::DimY, Window::Dimension(win_sub_plane2.y().start() / _y_subsampling[2], win_sub_plane2.y().end() / _y_subsampling[2], 1));

        unsigned int idx = 0;

        // Set inputs
        add_2D_tensor_argument(idx, _planes[0], slice);
        add_2D_tensor_argument(idx, _planes[1], win_sub_plane1);
        add_2D_tensor_argument(idx, _planes[2], win_sub_plane2);
        add_2D_tensor_argument_if((nullptr != _planes[3]), idx, _planes[3], slice);

        // Set outputs
        if(nullptr != _output) // Single planar output
        {
            add_2D_tensor_argument(idx, _output, slice);
        }
        else // Multi-planar output
        {
            // Reduce slice in case of subsampling to avoid out-of bounds access
            slice.set(Window::DimY, Window::Dimension(slice.y().start() / _y_subsampling[1], slice.y().end() / _y_subsampling[1], 1));

            add_2D_tensor_argument(idx, _output_multi->cl_plane(0), slice);
            add_2D_tensor_argument(idx, _output_multi->cl_plane(1), win_sub_plane1);
            add_2D_tensor_argument_if((3 == num_planes_from_format(_output_multi->info()->format())), idx, _output_multi->cl_plane(2), win_sub_plane2);

            _kernel.setArg(idx++, slice.y().end());
        }

        enqueue(queue, *this, slice);
    }
    while(window.slide_window_slice_2D(slice));
}
} // namespace arm_compute
