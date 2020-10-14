/*
 * Copyright (c) 2018-2020 Arm Limited.
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

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Utils.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, Window *output_window = nullptr)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);

    // Validate output if initialized
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
        if(output_window == nullptr)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(input->tensor_shape(), output->tensor_shape());
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(input->tensor_shape(), output_window->shape());
        }
    }

    return Status{};
}

} // namespace

CLCopyKernel::CLCopyKernel()
    : _input(nullptr), _output(nullptr), _output_window(), _has_output_window(false)
{
}

void CLCopyKernel::configure(const ICLTensor *input, ICLTensor *output, Window *output_window)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, output_window);
}

void CLCopyKernel::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, Window *output_window)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), output_window));

    auto padding_info = get_padding_info({ input, output });

    _input  = input;
    _output = output;

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*(output->info()), *(input->info()));

    // Configure window
    const unsigned int vec_size_x = adjust_vec_size(16 / input->info()->element_size(), input->info()->dimension(0));

    const Window win_config = calculate_max_window(*(input->info()), Steps(vec_size_x));

    if(output_window != nullptr)
    {
        _has_output_window             = true;
        _output_window                 = Window(*output_window);
        const int  width_x             = output_window->num_iterations(0);
        const int  vec_size_x_leftover = width_x % vec_size_x;
        const bool multi_access_x      = width_x >= static_cast<int32_t>(vec_size_x);

        if(multi_access_x)
        {
            _output_window.set(Window::DimX, Window::Dimension(output_window->x().start(), ceil_to_multiple(output_window->x().end(), vec_size_x), vec_size_x));
        }

        build_opts.add_option("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(vec_size_x_leftover));
    }
    else
    {
        const int width_x             = input->info()->tensor_shape().x();
        const int vec_size_x_leftover = width_x % vec_size_x;

        build_opts.add_option("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(vec_size_x_leftover));
    }

    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(vec_size_x));

    // Build kernel
    _kernel = create_kernel(compile_context, "copy_tensor", build_opts.options());

    // Validate and set the window
    ICLKernel::configure_internal(win_config);

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status CLCopyKernel::validate(const arm_compute::ITensorInfo *input, const arm_compute::ITensorInfo *output, Window *output_window)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, output_window));

    return Status{};
}

void CLCopyKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice;

    if(_has_output_window)
    {
        slice            = window.first_slice_window_3D();
        Window out_slice = _output_window.first_slice_window_3D();
        do
        {
            unsigned int idx = 0;
            add_3D_tensor_argument(idx, _input, slice);
            add_3D_tensor_argument(idx, _output, out_slice);
            enqueue(queue, *this, slice, lws_hint());
        }
        while(window.slide_window_slice_3D(slice) && _output_window.slide_window_slice_3D(out_slice));
    }
    else
    {
        Window collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
        slice            = collapsed.first_slice_window_3D();
        do
        {
            unsigned int idx = 0;
            add_3D_tensor_argument(idx, _input, slice);
            add_3D_tensor_argument(idx, _output, slice);
            enqueue(queue, *this, slice, lws_hint());
        }
        while(collapsed.slide_window_slice_3D(slice));
    }
}
} // namespace arm_compute
