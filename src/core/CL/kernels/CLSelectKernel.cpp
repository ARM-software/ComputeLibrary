/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLSelectKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Window.h"

#include "support/ToolchainSupport.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *c, const ITensorInfo *x, const ITensorInfo *y, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(x);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(x,
                                                         1,
                                                         DataType::U8, DataType::S8, DataType::QASYMM8,
                                                         DataType::U16, DataType::S16,
                                                         DataType::U32, DataType::S32,
                                                         DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(x, y);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(x, y);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(c, 1, DataType::U8);

    const bool is_same_rank = (c->tensor_shape().num_dimensions() == x->tensor_shape().num_dimensions());
    ARM_COMPUTE_RETURN_ERROR_ON(is_same_rank && (x->tensor_shape() != c->tensor_shape()));
    ARM_COMPUTE_RETURN_ERROR_ON(!is_same_rank && ((c->tensor_shape().num_dimensions() > 1) || (c->tensor_shape().x() != x->tensor_shape()[x->tensor_shape().num_dimensions() - 1])));

    if(output != nullptr && output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(x, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(x, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *c, ITensorInfo *x, ITensorInfo *y, ITensorInfo *output)
{
    if(output != nullptr)
    {
        // Output tensor auto initialization if not yet initialized
        auto_init_if_empty(*output, *x->clone());
    }

    const bool is_same_rank = (c->tensor_shape().num_dimensions() == x->tensor_shape().num_dimensions());

    const unsigned int num_elems_processed_per_iteration = 16 / x->element_size();

    // Configure kernel window
    Window                 win = calculate_max_window(*x, Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal x_access(x, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal y_access(y, 0, num_elems_processed_per_iteration);
    bool                   window_changed = update_window_and_padding(win, x_access, y_access);

    // Update window for condition
    if(is_same_rank)
    {
        AccessWindowHorizontal c_access(c, 0, num_elems_processed_per_iteration);
        window_changed = window_changed || update_window_and_padding(win, c_access);
    }

    // Update window for output
    if(output != nullptr)
    {
        AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
        window_changed = window_changed || update_window_and_padding(win, output_access);
        output_access.set_valid_region(win, x->valid_region());
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

CLSelectKernel::CLSelectKernel()
    : _c(nullptr), _x(nullptr), _y(nullptr), _output(nullptr), _has_same_rank(false)
{
}
void CLSelectKernel::configure(const ICLTensor *c, const ICLTensor *x, const ICLTensor *y, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(c, x, y, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(c->info(), x->info(), y->info(), output->info()));

    _c             = c;
    _x             = x;
    _y             = y;
    _output        = output;
    _has_same_rank = (c->info()->tensor_shape().num_dimensions() == x->info()->tensor_shape().num_dimensions());

    const unsigned int num_elems_processed_per_iteration = 16 / x->info()->element_size();

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(x->info()->data_type()));
    build_opts.add_option("-DSELECT_DATA_TYPE=" + get_cl_select_type_from_data_type(x->info()->data_type()));
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration));

    // Create kernel
    std::string kernel_name = "select";
    if(_has_same_rank)
    {
        kernel_name += "_same_rank";
    }
    else
    {
        const bool is_input_rank_greater_than_two = x->info()->tensor_shape().num_dimensions() > 2;
        if(is_input_rank_greater_than_two)
        {
            const size_t width      = x->info()->tensor_shape().x();
            const size_t height     = x->info()->tensor_shape().y();
            const size_t outer_size = x->info()->tensor_shape()[x->info()->tensor_shape().num_dimensions() - 1];
            const size_t depth_size = x->info()->tensor_shape().total_size() / (width * height * outer_size);
            build_opts.add_option("-DDEPTH_SIZE=" + support::cpp11::to_string(depth_size));
        }
        kernel_name += "_different_rank";
        kernel_name += is_input_rank_greater_than_two ? "_n" : "_2";
    }
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));

    // Configure kernel window
    auto win_config = validate_and_configure_window(c->info(), x->info(), y->info(), output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

    _config_id = "select_";
    _config_id += string_from_data_type(x->info()->data_type());
    _config_id += "_";
    _config_id += support::cpp11::to_string(x->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(x->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(x->info()->dimension(2));
}

Status CLSelectKernel::validate(const ITensorInfo *c, const ITensorInfo *x, const ITensorInfo *y, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(c, x, y, output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(c->clone().get(), x->clone().get(), y->clone().get(), output->clone().get()).first);
    return Status{};
}

void CLSelectKernel::run(const arm_compute::Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice     = collapsed.first_slice_window_3D();

    if(!_has_same_rank)
    {
        Window vector_slice = window.first_slice_window_1D();
        vector_slice.set(Window::DimX, Window::Dimension(0, 0, 0));
        unsigned int idx = 0;
        add_1D_tensor_argument(idx, _c, vector_slice);
    }

    do
    {
        unsigned int idx = _has_same_rank ? 0 : num_arguments_per_1D_tensor();
        if(_has_same_rank)
        {
            add_3D_tensor_argument(idx, _c, slice);
        }
        add_3D_tensor_argument(idx, _x, slice);
        add_3D_tensor_argument(idx, _y, slice);
        add_3D_tensor_argument(idx, _output, slice);

        enqueue(queue, *this, slice, lws_hint());
    }
    while(collapsed.slide_window_slice_3D(slice));
}
} // namespace arm_compute
