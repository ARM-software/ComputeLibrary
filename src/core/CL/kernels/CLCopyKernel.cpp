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
#include "arm_compute/core/CL/kernels/CLCopyKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const PaddingList &padding = PaddingList())
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(padding.size() > 4);

    // Validate output if initialized
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(misc::shape_calculator::compute_padded_shape(input->tensor_shape(), padding), output->tensor_shape());
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output, *input);

    // Configure window
    const unsigned int num_elems_processed_per_iteration = 16 / input->element_size();

    Window win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);

    bool window_changed = update_window_and_padding(win, input_access, output_access);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

std::pair<Status, Window> validate_and_configure_window_with_padding(ITensorInfo *input, ITensorInfo *output, const PaddingList &padding)
{
    TensorShape input_shape  = input->tensor_shape();
    TensorShape padded_shape = misc::shape_calculator::compute_padded_shape(input_shape, padding);

    auto_init_if_empty(*output, input->clone()->set_tensor_shape(padded_shape));

    // Configure window
    const unsigned int num_elems_processed_per_iteration = 16 / input->element_size();

    Window win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));

    // Pad on the x dimension accounting for the padding offset along the same dimension
    AccessWindowHorizontal output_access(output, padding[0].first, num_elems_processed_per_iteration);
    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
    bool                   window_changed = update_window_and_padding(win, input_access, output_access);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

/** Generate the string "-DPAD= @p dim @p index @p padding"
 *
 * @param[in] dim     The dimension index
 * @param[in] index   Can be 0 for the start dimension and 1 for the end dimension
 * @param[in] padding The value to pad for that index/dimension pair
 *
 * @return The correct concatenated string
 */
std::string generate_pad_string(const size_t dim, const size_t index, const size_t padding)
{
    return "-DPAD" + support::cpp11::to_string(dim) + support::cpp11::to_string(index) + "=" + support::cpp11::to_string(padding);
}

/** Pass the padding as build option to the kernel.
 *
 * @param[in]  tensor     The padded tensor
 * @param[in]  padding    The list of the padding for each dimension
 * @param[out] build_opts The build option to which adding the padding
 */
void add_padding_as_build_options(const PaddingList &padding, CLBuildOptions &build_opts)
{
    size_t dim = 0;
    for(dim = 0; dim < padding.size(); dim++)
    {
        build_opts.add_option(generate_pad_string(dim, 0, padding[dim].first));
        build_opts.add_option(generate_pad_string(dim, 1, padding[dim].second));
    }

    while(dim < TensorShape::num_max_dimensions)
    {
        build_opts.add_option(generate_pad_string(dim, 0, 0));
        build_opts.add_option(generate_pad_string(dim, 1, 0));
        dim++;
    }
}

} // namespace

CLCopyKernel::CLCopyKernel()
    : _input(nullptr), _output(nullptr)
{
}

void CLCopyKernel::configure(const ICLTensor *input, ICLTensor *output, const PaddingList &padding)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), padding));

    _input  = input;
    _output = output;

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));

    const unsigned int num_elems_processed_per_iteration = 16 / input->info()->element_size();
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration));

    std::pair<Status, Window> win_config;

    if(padding.empty())
    {
        // Build kernel
        _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("copy_tensor", build_opts.options()));

        // Configure window
        win_config = validate_and_configure_window(input->info(), output->info());
    }
    else
    {
        // Add compile time options
        add_padding_as_build_options(padding, build_opts);

        // If we are padding in the fourth dimension the kernel needs to know the depth of the
        // different cubes
        if(padding.size() == 4)
        {
            const size_t depth = input->info()->tensor_shape()[2];
            build_opts.add_option("-DDEPTH=" + support::cpp11::to_string(depth));
        }

        // Build kernel
        _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("copy_pad_tensor", build_opts.options()));

        // Configure window
        win_config = validate_and_configure_window_with_padding(input->info(), output->info(), padding);
    }

    // Validate and set the window
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);
}

Status CLCopyKernel::validate(const arm_compute::ITensorInfo *input, const arm_compute::ITensorInfo *output, const PaddingList &padding)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, padding));

    if(padding.empty())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get()).first);
    }
    else
    {
        ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_with_padding(input->clone().get(), output->clone().get(), padding).first);
    }

    return Status{};
}

void CLCopyKernel::run(const Window &window, cl::CommandQueue &queue)
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
        enqueue(queue, *this, slice);
    }
    while(collapsed.slide_window_slice_3D(slice));
}
} // namespace arm_compute
