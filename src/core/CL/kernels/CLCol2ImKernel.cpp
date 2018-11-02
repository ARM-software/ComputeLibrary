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
#include "arm_compute/core/CL/kernels/CLCol2ImKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include <cmath>

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, std::pair<unsigned int, unsigned int> convolved_dims)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QASYMM8, DataType::QS16, DataType::F16, DataType::F32);

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), compute_col2im_shape(*input, convolved_dims));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, std::pair<unsigned int, unsigned int> convolved_dims)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(compute_col2im_shape(*input, convolved_dims)));

    const unsigned int num_elems_read_per_iteration = is_data_type_fixed_point(input->data_type()) ? 1 : 8;

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

void CLCol2ImKernel::configure(const ICLTensor *input, ICLTensor *output, std::pair<unsigned int, unsigned int> convolved_dims)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), convolved_dims));

    _input          = input;
    _output         = output;
    _convolved_dims = convolved_dims;

    const DataType data_type = input->info()->data_type();

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type));
    build_opts.add_option("-DELEMENT_SIZE=" + support::cpp11::to_string(input->info()->element_size()));
    build_opts.add_option("-DWIDTH_INPUT=" + support::cpp11::to_string(input->info()->dimension(0)));
    build_opts.add_option("-DWIDTH_OUTPUT=" + support::cpp11::to_string(_convolved_dims.first));
    build_opts.add_option_if(is_data_type_fixed_point(data_type), "-DFIXED_POINT_POSITION=" + support::cpp11::to_string(input->info()->fixed_point_position()));

    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("col2im", build_opts.options()));

    // Configure the local work size for Bifrost with a value obtained
    // via exhaustive autotuning over 30 representative tensor shapes.
    const GPUTarget gpu_target = get_target();
    if(gpu_target_is_in(gpu_target, GPUTarget::G71, GPUTarget::G72, GPUTarget::G51, GPUTarget::G51BIG, GPUTarget::G51LIT, GPUTarget::TNOX))
    {
        if((_convolved_dims.first == 7) || (_convolved_dims.first == 14))
        {
            _lws_hint = cl::NDRange(1, 7, 1);
        }
        else
        {
            _lws_hint = cl::NDRange(1, 8, 1);
        }
    }

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), _convolved_dims);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure(win_config.second);

    // Set config_id for enabling LWS tuning
    _config_id = "col2im_";
    _config_id += lower_string(string_from_data_type(input->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(1));
}

Status CLCol2ImKernel::validate(const ITensorInfo *input, const ITensorInfo *output, std::pair<unsigned int, unsigned int> convolved_dims)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, convolved_dims));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(), convolved_dims).first);
    return Status{};
}

void CLCol2ImKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);
    // The collapse method rely on the assumption that the third dimension of input buffer is 1
    ARM_COMPUTE_ERROR_ON(window.z().end() != 1);

    Window collapsed_window = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice            = collapsed_window.first_slice_window_3D();

    // Set static kernel arguments
    unsigned int idx = 2 * num_arguments_per_3D_tensor();
    _kernel.setArg<cl_uint>(idx++, _output->info()->strides_in_bytes()[3]);

    do
    {
        // Set inputs
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice, _lws_hint);
    }
    while(collapsed_window.slide_window_slice_3D(slice));
}
