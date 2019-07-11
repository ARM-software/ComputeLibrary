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
#include "arm_compute/core/CL/kernels/CLMeanStdDevNormalizationKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, float epsilon)
{
    ARM_COMPUTE_UNUSED(epsilon);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->num_dimensions() > 2, "Input tensor cannot have more than 2 dimensions");
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);

    // Checks performed when output is configured
    if((output != nullptr) && (output->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }
    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    if(output != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
        // Output auto inizialitation if not yet initialized
        auto_init_if_empty(*output, *input);
    }

    const unsigned int num_elems_processed_per_iteration = 16 / input->element_size();

    // This kernel doesn't need padding
    Window win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
    if(output != nullptr)
    {
        output->set_valid_region(ValidRegion(Coordinates(), output->tensor_shape()));
    }

    return std::make_pair(Status{}, win);
}
} // namespace

CLMeanStdDevNormalizationKernel::CLMeanStdDevNormalizationKernel()
    : _input(nullptr), _output(nullptr), _run_in_place(false)
{
}

void CLMeanStdDevNormalizationKernel::configure(ICLTensor *input, ICLTensor *output, float epsilon)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input);

    _run_in_place = (output == nullptr) || (output == input);

    ARM_COMPUTE_ERROR_THROW_ON(CLMeanStdDevNormalizationKernel::validate(input->info(), (output != nullptr) ? output->info() : nullptr, epsilon));

    _input  = input;
    _output = output;

    const unsigned int num_elems_processed_per_iteration = 16 / input->info()->element_size();

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration));
    build_opts.add_option("-DEPSILON=" + float_to_string_with_full_precision(epsilon));
    build_opts.add_option("-DWIDTH=" + support::cpp11::to_string(input->info()->dimension(0)));
    build_opts.add_option_if(_run_in_place, "-DIN_PLACE");

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("mean_stddev_normalization", build_opts.options()));

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), (_run_in_place) ? nullptr : output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

    // Set config_id for enabling LWS tuning
    _config_id = "mean_stddev_normalization_layer_";
    _config_id += lower_string(string_from_data_type(input->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(1));
}

Status CLMeanStdDevNormalizationKernel::validate(const ITensorInfo *input, const ITensorInfo *output, float epsilon)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, epsilon));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), (output != nullptr) ? output->clone().get() : nullptr).first);
    return Status{};
}

void CLMeanStdDevNormalizationKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice = window.first_slice_window_2D();
    // Set slice step equal to width to force gws[0] to 1, as each thread normalizes across all rows
    slice.set_dimension_step(Window::DimX, _input->info()->dimension(0));

    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input, slice);
        add_2D_tensor_argument_if((!_run_in_place), idx, _output, slice);

        enqueue(queue, *this, slice, lws_hint());
    }
    while(window.slide_window_slice_2D(slice));
}
} // namespace arm_compute
