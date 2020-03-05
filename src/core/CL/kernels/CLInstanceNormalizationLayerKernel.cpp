/*
 * Copyright (c) 2019-2020 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLInstanceNormalizationLayerKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Window.h"

#include "support/StringSupport.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const InstanceNormalizationLayerKernelInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(info.epsilon == 0.f, "Epsilon must be different than 0");
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(input, DataType::F16, DataType::F32);

    if(output != nullptr && output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->num_channels() != output->num_channels(), "Input and output have different number of channels");
    }

    return Status{};
}

std::tuple<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    // We handle the planes manually
    Window win = calculate_max_window(*input, Steps(1));

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output, input->tensor_shape(), 1, input->data_type());

    // CLInstanceNormalizationLayerKernel doesn't need padding so update_window_and_padding() can be skipped
    Coordinates coord;
    coord.set_num_dimensions(output->num_dimensions());
    output->set_valid_region(ValidRegion(coord, output->tensor_shape()));
    return std::make_pair(Status{}, win);
}
} // namespace

CLInstanceNormalizationLayerKernel::CLInstanceNormalizationLayerKernel()
    : _input(nullptr), _output(nullptr), _run_in_place(false)
{
}

void CLInstanceNormalizationLayerKernel::configure(ICLTensor *input, ICLTensor *output, const InstanceNormalizationLayerKernelInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input);

    _input  = input;
    _output = output == nullptr ? input : output;

    _run_in_place = (output == nullptr) || (output == input);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(_input->info(), _output->info(), info));
    const unsigned int num_elems_processed_per_iteration = 16 / input->info()->element_size();

    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    build_opts.add_option("-DINTERNAL_DATA_TYPE=" + (info.use_mixed_precision ? "float" : get_cl_type_from_data_type(input->info()->data_type())));
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration));
    build_opts.add_option("-DDIM_X=" + support::cpp11::to_string(input->info()->dimension(0)));
    build_opts.add_option("-DDIM_Y=" + support::cpp11::to_string(input->info()->dimension(1)));
    build_opts.add_option("-DDIM_Z=" + support::cpp11::to_string(input->info()->dimension(2)));
    build_opts.add_option("-DGAMMA=" + float_to_string_with_full_precision(info.gamma));
    build_opts.add_option("-DBETA=" + float_to_string_with_full_precision(info.beta));
    build_opts.add_option("-DEPSILON=" + float_to_string_with_full_precision(info.epsilon));
    build_opts.add_option_if(_run_in_place, "-DIN_PLACE");
    build_opts.add_option_if(_input->info()->data_layout() == DataLayout::NHWC, "-DNHWC");

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("instance_normalization", build_opts.options()));

    // Configure kernel window
    auto win_config = validate_and_configure_window(_input->info(), _output->info());
    ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));
    ICLKernel::configure_internal(std::get<1>(win_config));
}

Status CLInstanceNormalizationLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const InstanceNormalizationLayerKernelInfo &info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, info));
    ARM_COMPUTE_RETURN_ON_ERROR(std::get<0>(validate_and_configure_window(input->clone().get(), (output == nullptr ? input->clone().get() : output->clone().get()))));
    return Status{};
}

void CLInstanceNormalizationLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    Window collapsed_window = window.collapse(window, Window::DimZ);

    // We will process the planes together
    if(_input->info()->data_layout() == DataLayout::NCHW)
    {
        collapsed_window.set(Window::DimX, Window::Dimension(0, 1, 1));
        collapsed_window.set(Window::DimY, Window::Dimension(0, 1, 1));
    }
    else
    {
        collapsed_window.set(Window::DimY, Window::Dimension(0, 1, 1));
        collapsed_window.set(Window::DimZ, Window::Dimension(0, _input->info()->dimension(3), 1));
    }

    unsigned int idx = 0;
    add_4D_tensor_argument(idx, _input, collapsed_window);
    if(!_run_in_place)
    {
        add_4D_tensor_argument(idx, _output, collapsed_window);
    }

    enqueue(queue, *this, collapsed_window, lws_hint());
}
} // namespace arm_compute
