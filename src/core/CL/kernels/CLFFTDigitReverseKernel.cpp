/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLFFTDigitReverseKernel.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *idx, const FFTDigitReverseKernelInfo &config)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() != DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_channels() != 1 && input->num_channels() != 2);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(idx, 1, DataType::U32);
    ARM_COMPUTE_RETURN_ERROR_ON(std::set<unsigned int>({ 0, 1 }).count(config.axis) == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(input->tensor_shape()[config.axis] != idx->tensor_shape().x());

    // Checks performed when output is configured
    if((output != nullptr) && (output->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON(output->num_channels() != 2);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, ITensorInfo *idx, const FFTDigitReverseKernelInfo &config)
{
    ARM_COMPUTE_UNUSED(idx, config);

    auto_init_if_empty(*output, input->clone()->set_num_channels(2));

    Window win = calculate_max_window(*output, Steps());
    output->set_valid_region(ValidRegion(Coordinates(), output->tensor_shape()));

    return std::make_pair(Status{}, win);
}
} // namespace

CLFFTDigitReverseKernel::CLFFTDigitReverseKernel()
    : _input(nullptr), _output(nullptr), _idx(nullptr)
{
}

void CLFFTDigitReverseKernel::configure(const ICLTensor *input, ICLTensor *output, const ICLTensor *idx, const FFTDigitReverseKernelInfo &config)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, idx);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), idx->info(), config));

    _input  = input;
    _output = output;
    _idx    = idx;

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(input->info()->num_channels()));
    build_opts.add_option_if(config.conjugate, "-DCONJ");
    std::string kernel_name = "fft_digit_reverse_axis_" + support::cpp11::to_string(config.axis);
    _kernel                 = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), idx->info(), config);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(input->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(1));
}

Status CLFFTDigitReverseKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *idx, const FFTDigitReverseKernelInfo &config)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, idx, config));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(), idx->clone().get(), config).first);

    return Status{};
}

void CLFFTDigitReverseKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice     = collapsed.first_slice_window_3D();

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _output, slice);
        add_1D_tensor_argument(idx, _idx, slice);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(collapsed.slide_window_slice_3D(slice));
}
} // namespace arm_compute
