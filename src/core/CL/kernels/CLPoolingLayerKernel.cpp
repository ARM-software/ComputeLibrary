/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLPoolingLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <set>
#include <string>
#include <tuple>

using namespace arm_compute;

CLPoolingLayerKernel::CLPoolingLayerKernel()
    : _input(nullptr), _output(nullptr), _pool_info(), _border_size(0), _num_elems_processed_per_iteration(1)
{
}

BorderSize CLPoolingLayerKernel::border_size() const
{
    return _border_size;
}

void CLPoolingLayerKernel::configure(const ICLTensor *input, ICLTensor *output, const PoolingLayerInfo &pool_info)
{
    int                 pool_pad_x      = 0;
    int                 pool_pad_y      = 0;
    int                 pool_stride_x   = 0;
    int                 pool_stride_y   = 0;
    unsigned int        pooled_w        = 0;
    unsigned int        pooled_h        = 0;
    const PoolingType   pool_type       = pool_info.pool_type();
    const int           pool_size       = pool_info.pool_size();
    const PadStrideInfo pad_stride_info = pool_info.pad_stride_info();
    std::tie(pool_pad_x, pool_pad_y)       = pad_stride_info.pad();
    std::tie(pool_stride_x, pool_stride_y) = pad_stride_info.stride();

    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_ERROR_ON(pool_pad_x >= pool_size || pool_pad_y >= pool_size);

    // Check output dimensions
    std::tie(pooled_w, pooled_h) = scaled_dimensions(input->info()->dimension(0),
                                                     input->info()->dimension(1),
                                                     pool_size,
                                                     pool_size,
                                                     pool_info.pad_stride_info());

    // Output auto initialization if not yet initialized
    {
        TensorShape output_shape{ input->info()->tensor_shape() };
        output_shape.set(0, pooled_w);
        output_shape.set(1, pooled_h);

        auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type(), input->info()->fixed_point_position());
    }

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);
    ARM_COMPUTE_ERROR_ON((output->info()->dimension(0) != pooled_w) || (output->info()->dimension(1) != pooled_h));

    const int input_width  = input->info()->dimension(0);
    const int input_height = input->info()->dimension(1);

    // Set instance variables
    _input       = input;
    _output      = output;
    _pool_info   = pool_info;
    _border_size = BorderSize(pool_pad_y, pool_pad_x);

    // Set build options
    std::set<std::string> build_opts;
    build_opts.emplace(("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type())));
    build_opts.emplace(("-DPOOL_" + string_from_pooling_type(pool_type)));
    if(is_data_type_fixed_point(input->info()->data_type()))
    {
        build_opts.emplace("-DFIXED_POINT_POSITION=" + support::cpp11::to_string(input->info()->fixed_point_position()));
    }

    build_opts.emplace(("-DSTRIDE_X=" + support::cpp11::to_string(pool_stride_x)));
    if(pool_type != PoolingType::MAX)
    {
        build_opts.emplace(("-DMAX_WIDTH=" + support::cpp11::to_string(input->info()->dimension(0) + pool_pad_x)));
        build_opts.emplace(("-DMAX_HEIGHT=" + support::cpp11::to_string(input->info()->dimension(1) + pool_pad_y)));
        build_opts.emplace(("-DSTRIDE_Y=" + support::cpp11::to_string(pool_stride_y)));
        build_opts.emplace(("-DPAD_X=" + support::cpp11::to_string(pool_pad_x)));
        build_opts.emplace(("-DPAD_Y=" + support::cpp11::to_string(pool_pad_y)));
    }

    // Create kernel
    if((pool_size == 2) || (pool_size == 3) || (pool_size == 7))
    {
        // Check if we have pool3x3 with stride_x less equal than 3. In these cases, run an optimized OpenCL kernel where
        // each thread computes 4 output elements
        const bool is_pool3x3_stride_le3 = (pool_size == 3) && (pool_stride_x <= 3) && !is_data_type_fixed_point(input->info()->data_type());

        int num_elements_read_per_iteration = (pool_size == 7) ? 8 : pool_size;
        if(is_pool3x3_stride_le3)
        {
            // Change the number of elements processed and number of elements read per iteration for pooling 3x3 with stride less equal than 3
            _num_elems_processed_per_iteration = 4;
            num_elements_read_per_iteration    = pool_size * (pool_stride_x + 1);
        }

        const int upper_bound_w = ((pooled_w - 1) * pool_stride_x - pool_pad_x + num_elements_read_per_iteration) - input_width;
        const int upper_bound_h = ((pooled_h - 1) * pool_stride_y - pool_pad_y + pool_size) - input_height;

        _border_size.right  = std::max(upper_bound_w, pool_pad_x);
        _border_size.bottom = std::max(upper_bound_h, pool_pad_y);

        std::string kernel_name = "pooling_layer_" + support::cpp11::to_string(pool_size);
        if(is_pool3x3_stride_le3)
        {
            _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name + "_optimized", build_opts));
        }
        else
        {
            _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts));
        }
    }
    else // Run general case
    {
        _num_elems_processed_per_iteration = 1;

        const int upper_bound_w = ((pooled_w - 1) * pool_stride_x - pool_pad_x + pool_size) - input_width;
        const int upper_bound_h = ((pooled_h - 1) * pool_stride_y - pool_pad_y + pool_size) - input_height;

        _border_size.right  = std::max(upper_bound_w, pool_pad_x);
        _border_size.bottom = std::max(upper_bound_h, pool_pad_y);

        build_opts.emplace(("-DPOOL_SIZE=" + support::cpp11::to_string(pool_size)));
        if(input->info()->data_type() == DataType::F16)
        {
            build_opts.emplace("-DFP16");
        }
        _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("pooling_layer_N", build_opts));
    }

    // Configure kernel window
    Window                 win = calculate_max_window(*output->info(), Steps(_num_elems_processed_per_iteration));
    AccessWindowStatic     input_access(input->info(), -pool_pad_x, -pool_pad_y, input_width + _border_size.right, input_height + _border_size.bottom);
    AccessWindowHorizontal output_access(output->info(), 0, _num_elems_processed_per_iteration);
    update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));
    ICLKernel::configure(win);
}

void CLPoolingLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    unsigned int pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y = 0;
    std::tie(pool_pad_x, pool_pad_y)       = _pool_info.pad_stride_info().pad();
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info().stride();

    Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice            = window_collapsed.first_slice_window_3D();

    do
    {
        // Upsample input by pool size
        Window in_slice(slice);
        in_slice.set(Window::DimX, Window::Dimension(in_slice.x().start() - pool_pad_x, in_slice.x().end() * pool_stride_x, pool_stride_x * _num_elems_processed_per_iteration));
        in_slice.set(Window::DimY, Window::Dimension(in_slice.y().start() - pool_pad_y, in_slice.y().end() * pool_stride_y, pool_stride_y));

        // Set inputs
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, in_slice);
        add_3D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice);
    }
    while(window_collapsed.slide_window_slice_3D(slice));
}
