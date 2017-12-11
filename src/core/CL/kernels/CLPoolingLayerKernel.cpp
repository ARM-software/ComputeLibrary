/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/core/CL/ICLKernel.h"
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

namespace
{
// Internal window config info
using CLPoolingConfig = std::pair<unsigned int, BorderSize>; //num_elems_processed_per_iteration, border_size

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
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((is_data_type_quantized_asymmetric(input->data_type()) && pool_info.pool_type() == PoolingType::L2),
                                    "Unsupported combination of parameters!");

    const bool         is_global_pooling = pool_info.is_global_pooling();
    const unsigned int pool_size         = is_global_pooling ? input->tensor_shape().x() : pool_info.pool_size();

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(is_global_pooling && (input->tensor_shape().x() != input->tensor_shape().y()),
                                    "Global pooling is supported only with rectangular inputs!");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!is_global_pooling && ((pool_info.pad_stride_info().pad().first >= pool_size) || (pool_info.pad_stride_info().pad().second >= pool_size)),
                                    "Invalid pool size and pool pad combination!");

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);

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

std::tuple<Status, Window, CLPoolingConfig> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, const PoolingLayerInfo &pool_info)
{
    int                 pool_pad_x      = 0;
    int                 pool_pad_y      = 0;
    int                 pool_stride_x   = 0;
    int                 pool_stride_y   = 0;
    unsigned int        pooled_w        = 0;
    unsigned int        pooled_h        = 0;
    int                 pool_size       = pool_info.pool_size();
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

    BorderSize     border_size = BorderSize(pool_pad_y, pool_pad_x);
    const DataType data_type   = input->data_type();

    const int input_width  = input->dimension(0);
    const int input_height = input->dimension(1);

    // Change the number of elements processed per iteration
    // for pooling 3x3 with stride less equal than 3
    const bool         can_optimize                      = (pool_size == 3) && (pool_stride_x <= 3) && !is_data_type_quantized(data_type);
    const unsigned int num_elems_processed_per_iteration = can_optimize ? 4 : 1;
    const int          num_elems_read_per_iteration      = (num_elems_processed_per_iteration - 1) * pool_stride_x + pool_size;

    // Number of iterations in X dimension
    const int num_iterations_x = (pooled_w + num_elems_processed_per_iteration - 1) / num_elems_processed_per_iteration;

    // Upper limit for the number of right/bottom border elements that are accessed
    const int upper_bound_w = ((num_iterations_x - 1) * num_elems_processed_per_iteration * pool_stride_x - pool_pad_x + num_elems_read_per_iteration) - input_width;
    const int upper_bound_h = ((pooled_h - 1) * pool_stride_y - pool_pad_y + pool_size) - input_height;

    border_size.right  = std::max(upper_bound_w, pool_pad_x);
    border_size.bottom = std::max(upper_bound_h, pool_pad_y);

    Window win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration));

    AccessWindowRectangle input_access(input, -pool_pad_x, -pool_pad_y, num_elems_read_per_iteration, pool_size,
                                       pool_stride_x * num_elems_processed_per_iteration, pool_stride_y);
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
    bool                   window_changed = update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_tuple(err, win, CLPoolingConfig(num_elems_processed_per_iteration, border_size));
}
} // namespace

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
    int                 pool_size       = pool_info.pool_size();
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

    const GPUTarget gpu_target = get_arch_from_target(get_target());
    const DataType  data_type  = input->info()->data_type();

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type));
    build_opts.add_option("-DPOOL_" + string_from_pooling_type(pool_type));
    build_opts.add_option_if(is_data_type_fixed_point(data_type),
                             "-DFIXED_POINT_POSITION=" + support::cpp11::to_string(input->info()->fixed_point_position()));
    build_opts.add_option("-DSTRIDE_X=" + support::cpp11::to_string(pool_stride_x));
    if(pool_type != PoolingType::MAX)
    {
        build_opts.add_option_if(exclude_padding, "-DEXCLUDE_PADDING");
        build_opts.add_option("-DMAX_WIDTH=" + support::cpp11::to_string(input->info()->dimension(0) + (exclude_padding ? 0 : pool_pad_x)));
        build_opts.add_option("-DMAX_HEIGHT=" + support::cpp11::to_string(input->info()->dimension(1) + (exclude_padding ? 0 : pool_pad_y)));
        build_opts.add_option("-DSTRIDE_Y=" + support::cpp11::to_string(pool_stride_y));
        build_opts.add_option("-DPAD_X=" + support::cpp11::to_string(pool_pad_x));
        build_opts.add_option("-DPAD_Y=" + support::cpp11::to_string(pool_pad_y));
    }

    // Create kernel
    if((pool_size == 3) && !is_data_type_quantized_asymmetric(data_type))
    {
        // Check if we have pool3x3 with stride_x less equal than 3. In these cases, run an optimized OpenCL kernel where
        // each thread computes 4 output elements
        const bool is_pool3x3_stride_le3 = (pool_size == 3) && (pool_stride_x <= 3) && !is_data_type_fixed_point(data_type);

        std::string kernel_name = ((is_pool3x3_stride_le3) ? "pooling_layer_optimized_" : "pooling_layer_")
                                  + support::cpp11::to_string(pool_size);
        _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));
    }
    else // Run general case
    {
        build_opts.add_option("-DPOOL_SIZE=" + support::cpp11::to_string(pool_size));
        build_opts.add_option_if(data_type == DataType::F16, "-DFP16");

        std::string kernel_name = is_data_type_quantized_asymmetric(data_type) ? "pooling_layer_N_quantized" : "pooling_layer_N";
        _kernel                 = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));
    }

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), pool_info);

    ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

    // Configure the local work size (hint) from the first two dimensions of the global work size.
    // On Bifrost, this works for up to 35x35xC filters, for which the pooling_layer_3_optimized
    // kernel is launched with gws=(9, 33, C). In any case, the hint will be ignored if it is
    // invalid (e.g. exceeds the maximum workgroup size that the kernel can be launched with).
    if(gpu_target == GPUTarget::BIFROST)
    {
        cl::NDRange gws = ICLKernel::gws_from_window(std::get<1>(win_config));
        _lws_hint       = cl::NDRange(gws[0], gws[1], 1);
    }

    ICLKernel::configure(std::get<1>(win_config));

    CLPoolingConfig pooling_config     = std::get<2>(win_config);
    _num_elems_processed_per_iteration = pooling_config.first;
    _border_size                       = pooling_config.second;

    // Set config_id for enabling LWS tuning
    _config_id = "pooling_layer_";
    _config_id += lower_string(string_from_data_type(data_type));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(1));
}

Status CLPoolingLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, pool_info));
    ARM_COMPUTE_RETURN_ON_ERROR(std::get<0>(validate_and_configure_window(input->clone().get(), output->clone().get(), pool_info)));

    return Status{};
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
        in_slice.set(Window::DimX, Window::Dimension(in_slice.x().start() - pool_pad_x,
                                                     (in_slice.x().end() - pool_pad_x) * pool_stride_x,
                                                     pool_stride_x * _num_elems_processed_per_iteration));
        in_slice.set(Window::DimY, Window::Dimension(in_slice.y().start() - pool_pad_y,
                                                     (in_slice.y().end() - pool_pad_y) * pool_stride_y,
                                                     pool_stride_y));

        // Set inputs
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, in_slice);
        add_3D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice, _lws_hint);
    }
    while(window_collapsed.slide_window_slice_3D(slice));
}
