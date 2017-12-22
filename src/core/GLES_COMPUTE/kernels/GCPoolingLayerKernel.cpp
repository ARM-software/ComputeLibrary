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
#include "arm_compute/core/GLES_COMPUTE/kernels/GCPoolingLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/GLES_COMPUTE/GCHelpers.h"
#include "arm_compute/core/GLES_COMPUTE/GCKernelLibrary.h"
#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"
#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <set>
#include <string>
#include <tuple>

using namespace arm_compute;

GCPoolingLayerKernel::GCPoolingLayerKernel()
    : _input(nullptr), _output(nullptr), _pool_info(), _border_size(0), _num_elems_processed_per_iteration(1)
{
}

BorderSize GCPoolingLayerKernel::border_size() const
{
    return _border_size;
}

void GCPoolingLayerKernel::configure(const IGCTensor *input, IGCTensor *output, const PoolingLayerInfo &pool_info)
{
    int                 pool_pad_x        = 0;
    int                 pool_pad_y        = 0;
    int                 pool_stride_x     = 0;
    int                 pool_stride_y     = 0;
    unsigned int        pooled_w          = 0;
    unsigned int        pooled_h          = 0;
    const PoolingType   pool_type         = pool_info.pool_type();
    int                 pool_size         = pool_info.pool_size();
    const PadStrideInfo pad_stride_info   = pool_info.pad_stride_info();
    const bool          is_global_pooling = pool_info.is_global_pooling();
    std::tie(pool_pad_x, pool_pad_y)       = pad_stride_info.pad();
    std::tie(pool_stride_x, pool_stride_y) = pad_stride_info.stride();

    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_ERROR_ON(!is_global_pooling && (pool_pad_x >= pool_size || pool_pad_y >= pool_size));
    ARM_COMPUTE_ERROR_ON(is_global_pooling && (input->info()->tensor_shape().x() != input->info()->tensor_shape().y()));

    // Update pool size in case of global pooling
    pool_size = is_global_pooling ? input->info()->dimension(0) : pool_size;

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
    build_opts.emplace("#define LOCAL_SIZE_X " + support::cpp11::to_string(1));
    build_opts.emplace("#define LOCAL_SIZE_Y " + support::cpp11::to_string(1));
    build_opts.emplace("#define LOCAL_SIZE_Z " + support::cpp11::to_string(1));
    if(input->info()->data_type() == DataType::F32)
    {
        build_opts.insert("#define DATA_TYPE_FP32");
    }
    else
    {
        build_opts.insert("#define DATA_TYPE_FP16");
    }
    build_opts.emplace(("#define POOL_" + string_from_pooling_type(pool_type)));
    build_opts.emplace(("#define STRIDE_X " + support::cpp11::to_string(pool_stride_x)));
    build_opts.emplace(("#define MAX_WIDTH " + support::cpp11::to_string(input->info()->dimension(0) + pool_pad_x)));
    build_opts.emplace(("#define MAX_HEIGHT " + support::cpp11::to_string(input->info()->dimension(1) + pool_pad_y)));
    build_opts.emplace(("#define STRIDE_Y " + support::cpp11::to_string(pool_stride_y)));
    build_opts.emplace(("#define PAD_X " + support::cpp11::to_string(pool_pad_x)));
    build_opts.emplace(("#define PAD_Y " + support::cpp11::to_string(pool_pad_y)));

    // Create kernel
    if((pool_size == 2) || (pool_size == 3) || (pool_size == 7))
    {
        // Check if we have pool3x3 with stride_x less equal than 3. In these cases, run an optimized OpenGLES kernel where
        // each thread computes 4 output elements
        const bool is_pool3x3_stride_le3 = (pool_size == 3) && (pool_stride_x <= 3) && !is_data_type_fixed_point(input->info()->data_type());

        int num_elements_read_per_iteration = (pool_size == 7) ? 8 : pool_size;

        if(input->info()->data_type() == DataType::F32)
        {
            if(is_pool3x3_stride_le3)
            {
                // Change the number of elements processed and number of elements read per iteration for pooling 3x3 with stride less equal than 3
                _num_elems_processed_per_iteration = 4;
                num_elements_read_per_iteration    = pool_size * (pool_stride_x + 1);
            }
        }
        else
        {
            num_elements_read_per_iteration = pool_size;
            if(is_pool3x3_stride_le3)
            {
                _num_elems_processed_per_iteration = 4;
            }
            else
            {
                _num_elems_processed_per_iteration = 2;
            }
        }

        const int upper_bound_w = ((pooled_w - 1) * pool_stride_x - pool_pad_x + num_elements_read_per_iteration) - input_width;
        const int upper_bound_h = ((pooled_h - 1) * pool_stride_y - pool_pad_y + pool_size) - input_height;

        _border_size.right  = std::max(upper_bound_w, pool_pad_x);
        _border_size.bottom = std::max(upper_bound_h, pool_pad_y);

        std::string kernel_name = "pooling_layer_" + support::cpp11::to_string(pool_size);
        if(is_pool3x3_stride_le3)
        {
            build_opts.insert("#define POOLING_LAYER_3_OPTIMIZED");
            _kernel = static_cast<GCKernel>(GCKernelLibrary::get().create_kernel(kernel_name + "_optimized", build_opts));
        }
        else
        {
            build_opts.insert("#define POOLING_LAYER_" + support::cpp11::to_string(pool_size));
            _kernel = static_cast<GCKernel>(GCKernelLibrary::get().create_kernel(kernel_name, build_opts));
        }
    }
    else // Run general case
    {
        if(input->info()->data_type() == DataType::F32)
        {
            _num_elems_processed_per_iteration = 1;
        }
        else
        {
            _num_elems_processed_per_iteration = 2;
        }
        const int upper_bound_w = ((pooled_w - 1) * pool_stride_x - pool_pad_x + pool_size) - input_width;
        const int upper_bound_h = ((pooled_h - 1) * pool_stride_y - pool_pad_y + pool_size) - input_height;

        _border_size.right  = std::max(upper_bound_w, pool_pad_x);
        _border_size.bottom = std::max(upper_bound_h, pool_pad_y);

        build_opts.emplace(("#define POOL_SIZE " + support::cpp11::to_string(pool_size)));

        build_opts.insert("#define POOLING_LAYER_N");
        _kernel = static_cast<GCKernel>(GCKernelLibrary::get().create_kernel("pooling_layer_n", build_opts));
    }

    Window win = calculate_max_window(*output->info(), Steps(_num_elems_processed_per_iteration));

    if(input->info()->data_type() == DataType::F32)
    {
        AccessWindowStatic     input_access(input->info(), -pool_pad_x, -pool_pad_y, input_width + _border_size.right, input_height + _border_size.bottom);
        AccessWindowHorizontal output_access(output->info(), 0, _num_elems_processed_per_iteration);
        update_window_and_padding(win, input_access, output_access);
        output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));
    }
    else
    {
        // Calculate output right and bottom border
        const int output_width          = output->info()->dimension(0);
        const int output_height         = output->info()->dimension(1);
        const int output_padding_right  = ceil_to_multiple(output_width, _num_elems_processed_per_iteration) - output_width;
        const int output_padding_bottom = ceil_to_multiple(output_height, 1) - output_height;
        const int input_padding_right   = ceil_to_multiple(input_width + 2 * _border_size.right, _num_elems_processed_per_iteration) - (input_width + 2 * _border_size.right);
        const int input_padding_bottom  = ceil_to_multiple(input_height + 2 * _border_size.bottom, 1) - (input_height + 2 * _border_size.bottom);

        // Configure kernel window
        AccessWindowStatic input_access(input->info(), -pool_pad_x, -pool_pad_y, input_width + _border_size.right + input_padding_right, input_height + _border_size.bottom + input_padding_bottom);
        AccessWindowStatic output_access(output->info(), 0, 0, output_width + output_padding_right, output_height + output_padding_bottom);
        update_window_and_padding(win, input_access, output_access);
        output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));
    }

    IGCKernel::configure(win);
}

void GCPoolingLayerKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    unsigned int pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y = 0;
    std::tie(pool_pad_x, pool_pad_y)       = _pool_info.pad_stride_info().pad();
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info().stride();

    _kernel.use();

    Window window_collapsed = window.collapse_if_possible(IGCKernel::window(), Window::DimZ);
    Window slice            = window_collapsed.first_slice_window_3D();

    do
    {
        // Upsample input by pool size
        Window in_slice(slice);
        in_slice.set(Window::DimX, Window::Dimension(in_slice.x().start() - pool_pad_x, in_slice.x().end() * pool_stride_x, pool_stride_x * _num_elems_processed_per_iteration));
        in_slice.set(Window::DimY, Window::Dimension(in_slice.y().start() - pool_pad_y, in_slice.y().end() * pool_stride_y, pool_stride_y));

        // Set inputs
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, 1, in_slice);
        add_3D_tensor_argument(idx, _output, 2, slice);

        _kernel.update_shader_params();
        enqueue(*this, slice);
    }
    while(window_collapsed.slide_window_slice_3D(slice));
}
