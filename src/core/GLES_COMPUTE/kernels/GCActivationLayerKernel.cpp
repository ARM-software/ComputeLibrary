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
#include "arm_compute/core/GLES_COMPUTE/kernels/GCActivationLayerKernel.h"

#include "arm_compute/core/GLES_COMPUTE/GCHelpers.h"
#include "arm_compute/core/GLES_COMPUTE/GCKernelLibrary.h"
#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "support/ToolchainSupport.h"

#include <set>
#include <string>

using namespace arm_compute;

GCActivationLayerKernel::GCActivationLayerKernel()
    : _input(nullptr), _output(nullptr)
{
}

void GCActivationLayerKernel::configure(IGCTensor *input, IGCTensor *output, ActivationLayerInfo act_info)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);

    // Make sure _kernel is initialized before calling the parent's configure
    _input  = input;
    _output = input;

    if(output != nullptr)
    {
        // Output auto inizialitation if not yet initialized
        auto_init_if_empty(*output->info(), input->info()->tensor_shape(), 1, input->info()->data_type());

        ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

        _output = output;
    }

    unsigned int num_elems_processed_per_iteration = 4 / input->info()->element_size();

    // Set build options
    std::set<std::string> build_opts;
    std::string           dt_name = (input->info()->data_type() == DataType::F32) ? "DATA_TYPE_FP32" : "DATA_TYPE_FP16";
    build_opts.emplace(("#define " + string_from_activation_func(act_info.activation())));
    build_opts.emplace(("#define " + dt_name));
    build_opts.emplace(("#define A_VAL " + float_to_string_with_full_precision(act_info.a())));
    build_opts.emplace(("#define B_VAL " + float_to_string_with_full_precision(act_info.b())));
    build_opts.emplace(("#define LOCAL_SIZE_X " + support::cpp11::to_string(1)));
    build_opts.emplace(("#define LOCAL_SIZE_Y " + support::cpp11::to_string(1)));
    build_opts.emplace(("#define LOCAL_SIZE_Z " + support::cpp11::to_string(1)));

    // Create kernel
    _kernel = static_cast<GCKernel>(GCKernelLibrary::get().create_kernel("activation_layer", build_opts));

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    if(output != nullptr)
    {
        AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

        update_window_and_padding(win,
                                  AccessWindowHorizontal(input->info(), 0, num_elems_processed_per_iteration),
                                  output_access);

        output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));
    }
    else
    {
        update_window_and_padding(win,
                                  AccessWindowHorizontal(input->info(), 0, num_elems_processed_per_iteration));
    }

    IGCKernel::configure(win);
}

void GCActivationLayerKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IGCKernel::window(), window);

    _kernel.use();

    _output->set_needs_shifting(true);

    Window collapsed = window.collapse_if_possible(IGCKernel::window(), Window::DimZ);
    Window slice     = collapsed.first_slice_window_3D();
    Window slice_in  = collapsed.first_slice_window_3D();

    slice.shift(Window::DimX, -(_output->info()->padding()).left);

    if(_input == _output)
    {
        slice_in.shift(Window::DimX, -(_input->info()->padding()).left);
    }

    do
    {
        unsigned int idx     = 0;
        unsigned int binding = 1;
        add_3D_tensor_argument(idx, _input, binding++, slice);
        add_3D_tensor_argument(idx, _output, binding++, slice_in);
        _kernel.update_shader_params();
        enqueue(*this, slice);
    }
    while(collapsed.slide_window_slice_3D(slice) && collapsed.slide_window_slice_3D(slice_in));
}
