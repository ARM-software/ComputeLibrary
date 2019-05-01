/*
 * Copyright (c) 2017-2019 ARM Limited.
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

namespace
{
// Internal window config info
using GCPoolingConfig = std::pair<unsigned int, BorderSize>; //num_elems_processed_per_iteration, border_size

void auto_init(const ITensorInfo *input, ITensorInfo *output, unsigned int pooled_w, unsigned int pooled_h)
{
    TensorShape output_shape{ input->tensor_shape() };
    output_shape.set(0, pooled_w);
    output_shape.set(1, pooled_h);

    auto_init_if_empty(*output, input->clone()->set_tensor_shape(output_shape));
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((is_data_type_quantized_asymmetric(input->data_type()) && pool_info.pool_type() == PoolingType::L2),
                                    "Unsupported combination of parameters!");
    ARM_COMPUTE_RETURN_ERROR_ON(!pool_info.pad_stride_info().padding_is_symmetric());

    const bool         is_global_pooling = pool_info.is_global_pooling();
    const unsigned int pool_size         = is_global_pooling ? input->tensor_shape().x() : pool_info.pool_size().width;

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(is_global_pooling && (input->tensor_shape().x() != input->tensor_shape().y()),
                                    "Global pooling is supported only with rectangular inputs!");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!is_global_pooling && ((pool_info.pad_stride_info().pad().first >= pool_size) || (pool_info.pad_stride_info().pad().second >= pool_size)),
                                    "Invalid pool size and pool pad combination!");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(pool_info.pool_size().width != pool_info.pool_size().height, "Invalid Pool size, width not equal to height!");

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

        unsigned int pooled_w = 0;
        unsigned int pooled_h = 0;
        std::tie(pooled_w, pooled_h) = scaled_dimensions(input->dimension(0),
                                                         input->dimension(1),
                                                         pool_size,
                                                         pool_size,
                                                         pool_info.pad_stride_info());
        ARM_COMPUTE_RETURN_ERROR_ON_MSG((output->dimension(0) != pooled_w) || (output->dimension(1) != pooled_h),
                                        "Invalid output pooling dimensions!");
    }

    return Status{};
}

std::tuple<Status, Window, GCPoolingConfig> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, const PoolingLayerInfo &pool_info)
{
    int                 pool_pad_x      = 0;
    int                 pool_pad_y      = 0;
    int                 pool_stride_x   = 0;
    int                 pool_stride_y   = 0;
    unsigned int        pooled_w        = 0;
    unsigned int        pooled_h        = 0;
    int                 pool_size       = pool_info.pool_size().width;
    const PadStrideInfo pad_stride_info = pool_info.pad_stride_info();
    std::tie(pool_pad_x, pool_pad_y)       = pad_stride_info.pad();
    std::tie(pool_stride_x, pool_stride_y) = pad_stride_info.stride();

    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Update pool size in case of global pooling
    pool_size = pool_info.is_global_pooling() ? input->dimension(0) : pool_size;

    // Check output dimensions
    std::tie(pooled_w, pooled_h) = scaled_dimensions(input->dimension(0),
                                                     input->dimension(1),
                                                     pool_size,
                                                     pool_size,
                                                     pad_stride_info);

    auto_init(input, output, pooled_w, pooled_h);

    BorderSize border_size = BorderSize(pool_pad_y, pool_pad_x);

    const int input_width  = input->dimension(0);
    const int input_height = input->dimension(1);

    unsigned int num_elems_processed_per_iteration = 1;

    // Create kernel
    if(pool_size == 3)
    {
        // Check if we have pool3x3 with stride_x less equal than 3. In these cases, run an optimized OpenGLES kernel where
        // each thread computes 4 output elements
        const bool is_pool3x3_stride_le3 = (pool_size == 3) && (pool_stride_x <= 3);

        int num_elems_read_per_iteration = pool_size;

        if(input->data_type() == DataType::F32)
        {
            if(is_pool3x3_stride_le3)
            {
                // Change the number of elements processed and number of elements read per iteration for pooling 3x3 with stride less equal than 3
                num_elems_processed_per_iteration = 4;
                num_elems_read_per_iteration      = pool_size * (pool_stride_x + 1);
            }
        }
        else
        {
            if(is_pool3x3_stride_le3)
            {
                num_elems_processed_per_iteration = 4;
            }
            else
            {
                num_elems_processed_per_iteration = 2;
            }
        }

        const int upper_bound_w = ((pooled_w - 1) * pool_stride_x - pool_pad_x + num_elems_read_per_iteration) - input_width;
        const int upper_bound_h = ((pooled_h - 1) * pool_stride_y - pool_pad_y + pool_size) - input_height;

        border_size.right  = std::max(upper_bound_w, pool_pad_x);
        border_size.bottom = std::max(upper_bound_h, pool_pad_y);
    }
    else // Run general case
    {
        if(input->data_type() == DataType::F32)
        {
            num_elems_processed_per_iteration = 1;
        }
        else
        {
            num_elems_processed_per_iteration = 2;
        }

        const int upper_bound_w = ((pooled_w - 1) * pool_stride_x - pool_pad_x + pool_size) - input_width;
        const int upper_bound_h = ((pooled_h - 1) * pool_stride_y - pool_pad_y + pool_size) - input_height;

        border_size.right  = std::max(upper_bound_w, pool_pad_x);
        border_size.bottom = std::max(upper_bound_h, pool_pad_y);
    }
    // Configure kernel window
    Window win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration));

    if(input->data_type() == DataType::F32)
    {
        AccessWindowStatic     input_access(input, -pool_pad_x, -pool_pad_y, input_width + border_size.right, input_height + border_size.bottom);
        AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
        bool                   window_changed = update_window_and_padding(win, input_access, output_access);
        output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));
        Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
        return std::make_tuple(err, win, GCPoolingConfig(num_elems_processed_per_iteration, border_size));
    }
    else
    {
        // Calculate output right and bottom border
        const int output_width          = output->dimension(0);
        const int output_height         = output->dimension(1);
        const int output_padding_right  = ceil_to_multiple(output_width, num_elems_processed_per_iteration) - output_width;
        const int output_padding_bottom = ceil_to_multiple(output_height, 1) - output_height;

        const int input_total_width    = std::max(int(input->padding().left), int(pool_pad_x)) + input_width + std::max(int(input->padding().right), int(pool_pad_x));
        const int input_padding_right  = ceil_to_multiple(input_total_width, num_elems_processed_per_iteration) - input_width - pool_pad_x;
        const int input_total_height   = std::max(int(input->padding().top), int(pool_pad_y)) + input_height + std::max(int(input->padding().bottom), int(pool_pad_y));
        const int input_padding_bottom = input_total_height - input_height - pool_pad_y;

        // Configure kernel window
        AccessWindowStatic input_access(input, -pool_pad_x, -pool_pad_y, input_width + input_padding_right, input_height + input_padding_bottom);
        AccessWindowStatic output_access(output, 0, 0, output_width + output_padding_right, output_height + output_padding_bottom);
        bool               window_changed = update_window_and_padding(win, input_access, output_access);
        output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));
        Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
        return std::make_tuple(err, win, GCPoolingConfig(num_elems_processed_per_iteration, border_size));
    }
}
} // namespace

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
    int                 pool_pad_x      = 0;
    int                 pool_pad_y      = 0;
    int                 pool_stride_x   = 0;
    int                 pool_stride_y   = 0;
    unsigned int        pooled_w        = 0;
    unsigned int        pooled_h        = 0;
    const PoolingType   pool_type       = pool_info.pool_type();
    int                 pool_size       = pool_info.pool_size().width;
    const PadStrideInfo pad_stride_info = pool_info.pad_stride_info();
    const bool          exclude_padding = pool_info.exclude_padding();
    std::tie(pool_pad_x, pool_pad_y)       = pad_stride_info.pad();
    std::tie(pool_stride_x, pool_stride_y) = pad_stride_info.stride();

    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Update pool size in case of global pooling
    pool_size = pool_info.is_global_pooling() ? input->info()->dimension(0) : pool_size;

    // Check output dimensions
    std::tie(pooled_w, pooled_h) = scaled_dimensions(input->info()->dimension(0),
                                                     input->info()->dimension(1),
                                                     pool_size,
                                                     pool_size,
                                                     pad_stride_info);

    auto_init(input->info(), output->info(), pooled_w, pooled_h);

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), pool_info));

    // Set instance variables
    _input     = input;
    _output    = output;
    _pool_info = pool_info;

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
    if(exclude_padding)
    {
        build_opts.emplace("#define EXCLUDE_PADDING");
    }
    build_opts.emplace(("#define POOL_" + string_from_pooling_type(pool_type)));
    build_opts.emplace(("#define STRIDE_X " + support::cpp11::to_string(pool_stride_x)));
    build_opts.emplace(("#define MAX_WIDTH " + support::cpp11::to_string(input->info()->dimension(0) + (exclude_padding ? 0 : pool_pad_x))));
    build_opts.emplace(("#define MAX_HEIGHT " + support::cpp11::to_string(input->info()->dimension(1) + (exclude_padding ? 0 : pool_pad_y))));
    build_opts.emplace(("#define STRIDE_Y " + support::cpp11::to_string(pool_stride_y)));
    build_opts.emplace(("#define PAD_X " + support::cpp11::to_string(pool_pad_x)));
    build_opts.emplace(("#define PAD_Y " + support::cpp11::to_string(pool_pad_y)));

    // Create kernel
    if((pool_size == 2) || (pool_size == 3) || (pool_size == 7))
    {
        // Check if we have pool3x3 with stride_x less equal than 3. In these cases, run an optimized OpenGLES kernel where
        // each thread computes 4 output elements
        const bool is_pool3x3_stride_le3 = (pool_size == 3) && (pool_stride_x <= 3);

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
        build_opts.emplace(("#define POOL_SIZE " + support::cpp11::to_string(pool_size)));

        build_opts.insert("#define POOLING_LAYER_N");
        _kernel = static_cast<GCKernel>(GCKernelLibrary::get().create_kernel("pooling_layer_n", build_opts));
    }
    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), pool_info);
    ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

    IGCKernel::configure(std::get<1>(win_config));
    GCPoolingConfig pooling_config     = std::get<2>(win_config);
    _num_elems_processed_per_iteration = pooling_config.first;
    _border_size                       = pooling_config.second;
}

Status GCPoolingLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, pool_info));
    ARM_COMPUTE_RETURN_ON_ERROR(std::get<0>(validate_and_configure_window(input->clone().get(), output->clone().get(), pool_info)));

    return Status{};
}

void GCPoolingLayerKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    unsigned int pool_pad_x;
    unsigned int pool_pad_y;
    unsigned int pool_stride_x;
    unsigned int pool_stride_y;
    std::tie(pool_pad_x, pool_pad_y)       = _pool_info.pad_stride_info().pad();
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info().stride();

    _kernel.use();

    _output->set_needs_shifting(true);

    Window window_collapsed = window.collapse_if_possible(IGCKernel::window(), Window::DimZ);

    Window slice         = window_collapsed.first_slice_window_3D();
    Window slice_in_orig = window_collapsed.first_slice_window_3D();

    slice.shift(Window::DimX, -(_output->info()->padding()).left);

    do
    {
        // Upsample input by pool size
        Window in_slice(slice_in_orig); // NOLINT
        in_slice.set(Window::DimX, Window::Dimension(in_slice.x().start() - pool_pad_x, in_slice.x().end() * pool_stride_x, pool_stride_x * _num_elems_processed_per_iteration));
        in_slice.set(Window::DimY, Window::Dimension(in_slice.y().start() - pool_pad_y, in_slice.y().end() * pool_stride_y, pool_stride_y));

        // Set inputs
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, 1, in_slice);
        add_3D_tensor_argument(idx, _output, 2, slice);

        _kernel.update_shader_params();
        enqueue(*this, slice);
    }
    while(window_collapsed.slide_window_slice_3D(slice) && window_collapsed.slide_window_slice_3D(slice_in_orig));
}
