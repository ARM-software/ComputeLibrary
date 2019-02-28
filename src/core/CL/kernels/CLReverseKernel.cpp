/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLReverseKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Window.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *axis)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, axis);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S8, DataType::QASYMM8,
                                                         DataType::U16, DataType::S16,
                                                         DataType::U32, DataType::S32,
                                                         DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(axis, 1, DataType::U32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis->num_dimensions() > 1, "Axis must be a 1D tensor");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis->dimension(0) > 4, "Only up to 4 dimensions can be reversed");

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    }

    return Status{};
}
} // namespace

CLReverseKernel::CLReverseKernel()
    : _input(nullptr), _output(nullptr), _axis(nullptr)
{
}

void CLReverseKernel::configure(const ICLTensor *input, ICLTensor *output, const ICLTensor *axis)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, axis);

    _input  = input;
    _output = output;
    _axis   = axis;

    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), *input->info()->clone());

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), axis->info()));

    // Set kernel build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DNUM_REVERSE_DIMS=" + support::cpp11::to_string(axis->info()->dimension(0)));
    switch(input->info()->element_size())
    {
        case 1:
            build_opts.add_option("-DDATA_TYPE=uchar");
            break;
        case 2:
            build_opts.add_option("-DDATA_TYPE=ushort");
            break;
        case 4:
            build_opts.add_option("-DDATA_TYPE=uint");
            break;
        default:
            ARM_COMPUTE_ERROR("Data type not supported");
    }

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("reverse", build_opts.options()));

    // Set static kernel arguments
    unsigned int idx = 2 * num_arguments_per_4D_tensor() + num_arguments_per_1D_tensor();
    add_argument<cl_uint>(idx, input->info()->dimension(0));
    add_argument<cl_uint>(idx, input->info()->dimension(1));
    add_argument<cl_uint>(idx, input->info()->dimension(2));
    add_argument<cl_uint>(idx, input->info()->dimension(3));

    // Configure kernel window
    Window win = calculate_max_window(*output->info(), Steps());
    ICLKernel::configure_internal(win);

    // Set config_id for enabling LWS tuning
    _config_id += "reverse_";
    _config_id += lower_string(string_from_data_type(input->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(2));
}

Status CLReverseKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *axis)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, axis));
    return Status{};
}

void CLReverseKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window collapsed  = window.collapse(ICLKernel::window(), Window::DimZ);
    Window slice      = collapsed.first_slice_window_4D();
    Window axis_slice = collapsed.first_slice_window_1D();

    do
    {
        unsigned int idx = 0;
        add_4D_tensor_argument(idx, _input, slice);
        add_1D_tensor_argument(idx, _axis, axis_slice);
        add_4D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(collapsed.slide_window_slice_4D(slice));
}
} // namespace arm_compute
