/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLActivationLayerKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/FixedPoint.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "support/ToolchainSupport.h"

#include <cmath>

using namespace arm_compute;

CLActivationLayerKernel::CLActivationLayerKernel()
    : _input(nullptr), _output(nullptr)
{
}

void CLActivationLayerKernel::configure(ICLTensor *input, ICLTensor *output, ActivationLayerInfo act_info)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);

    if(output != nullptr)
    {
        // Output auto inizialitation if not yet initialized
        auto_init_if_empty(*output->info(), input->info()->tensor_shape(), 1, input->info()->data_type(), input->info()->fixed_point_position());

        ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);
    }

    const unsigned int num_elems_processed_per_iteration = 16 / input->info()->element_size();
    const int          fixed_point_position              = input->info()->fixed_point_position();
    float              a_const                           = act_info.a();
    float              b_const                           = act_info.b();
    if(is_data_type_fixed_point(input->info()->data_type()))
    {
        a_const = static_cast<int>(lround(a_const * (1 << fixed_point_position)));
        b_const = static_cast<int>(lround(b_const * (1 << fixed_point_position)));
    }

    // Set build options
    std::set<std::string> build_opts;
    build_opts.emplace(("-DACT=" + lower_string(string_from_activation_func(act_info.activation()))));
    build_opts.emplace(("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type())));
    build_opts.emplace(("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration)));
    build_opts.emplace(("-DA_VAL=" + support::cpp11::to_string(a_const)));
    build_opts.emplace(("-DB_VAL=" + support::cpp11::to_string(b_const)));
    build_opts.emplace(output == nullptr ? "-DIN_PLACE" : "");
    if(is_data_type_fixed_point(input->info()->data_type()))
    {
        build_opts.emplace(("-DFIXED_POINT_POSITION=" + support::cpp11::to_string(fixed_point_position)));
    }

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("activation_layer", build_opts));

    // Make sure _kernel is initialized before calling the parent's configure

    _input  = input;
    _output = output;

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    if(output != nullptr)
    {
        AccessWindowHorizontal input_access(input->info(), 0, num_elems_processed_per_iteration);
        AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);
        update_window_and_padding(win, input_access, output_access);
        output_access.set_valid_region(win, input->info()->valid_region());
    }
    else
    {
        update_window_and_padding(win,
                                  AccessWindowHorizontal(input->info(), 0, num_elems_processed_per_iteration));
    }

    ICLKernel::configure(win);
}

void CLActivationLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice     = collapsed.first_slice_window_3D();

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        if(_output != nullptr)
        {
            add_3D_tensor_argument(idx, _output, slice);
        }
        enqueue(queue, *this, slice);
    }
    while(collapsed.slide_window_slice_3D(slice));
}
