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
#include "arm_compute/core/CL/kernels/CLCol2ImKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include <cmath>

using namespace arm_compute::misc::shape_calculator;

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const Size2D &convolved_dims, unsigned int num_groups)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), compute_col2im_shape(*input, convolved_dims, true, num_groups));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->data_layout() != DataLayout::NCHW, "Col2Im output's data layout must always be NCHW");
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, const Size2D &convolved_dims, unsigned int num_groups)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(compute_col2im_shape(*input, convolved_dims, true, num_groups)).set_data_layout(DataLayout::NCHW));

    constexpr unsigned int num_elems_read_per_iteration = 8;

    // Configure window
    Window win = calculate_max_window(*input, Steps(num_elems_read_per_iteration));

    // Update window and padding just for the input tensor as we cannot access out-of-bounds elements in the output one
    AccessWindowHorizontal input_access(input, 0, num_elems_read_per_iteration);
    bool                   window_changed = update_window_and_padding(win, input_access);

    Coordinates coord;
    coord.set_num_dimensions(output->num_dimensions());
    output->set_valid_region(ValidRegion(coord, output->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

CLCol2ImKernel::CLCol2ImKernel()
    : _input(nullptr), _output(nullptr), _convolved_dims()
{
}

void CLCol2ImKernel::configure(const ICLTensor *input, ICLTensor *output, const Size2D &convolved_dims, unsigned int num_groups)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), convolved_dims, num_groups));

    _input          = input;
    _output         = output;
    _convolved_dims = convolved_dims;

    const DataType data_type = input->info()->data_type();

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type));
    build_opts.add_option("-DELEMENT_SIZE=" + support::cpp11::to_string(input->info()->element_size()));
    build_opts.add_option("-DWIDTH_INPUT=" + support::cpp11::to_string(input->info()->dimension(0)));
    build_opts.add_option("-DWIDTH_OUTPUT=" + support::cpp11::to_string(_convolved_dims.width));
    build_opts.add_option("-DNUM_GROUPS=" + support::cpp11::to_string(num_groups));

    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("col2im", build_opts.options()));

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), _convolved_dims, num_groups);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

    // Set config_id for enabling LWS tuning
    _config_id = "col2im_";
    _config_id += lower_string(string_from_data_type(input->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(num_groups);
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(1));
}

Status CLCol2ImKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const Size2D &convolved_dims, unsigned int num_groups)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, convolved_dims, num_groups));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(), convolved_dims, num_groups).first);
    return Status{};
}

void CLCol2ImKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

    bool is_collapsed     = false;
    bool is_collapsed_out = false;

    Window out_window;
    out_window.use_tensor_dimensions(_output->info()->tensor_shape());

    Window collapsed     = window.collapse_if_possible(ICLKernel::window(), Window::DimZ, &is_collapsed);
    Window collapsed_out = out_window.collapse_if_possible(out_window, 3, &is_collapsed_out);

    ARM_COMPUTE_ERROR_ON(is_collapsed != is_collapsed_out);

    Window slice     = collapsed.first_slice_window_3D();
    Window slice_out = collapsed_out.first_slice_window_4D();
    do
    {
        // Set inputs
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        add_4D_tensor_argument(idx, _output, slice_out);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(collapsed.slide_window_slice_3D(slice) && collapsed_out.slide_window_slice_4D(slice_out));
}
} // namespace arm_compute
