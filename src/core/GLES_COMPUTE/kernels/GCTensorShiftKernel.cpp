/*
 * Copyright (c) 2017-2020 ARM Limited.
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
#include "arm_compute/core/GLES_COMPUTE/kernels/GCTensorShiftKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/GLES_COMPUTE/GCHelpers.h"
#include "arm_compute/core/GLES_COMPUTE/GCKernelLibrary.h"
#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "support/StringSupport.h"

using namespace arm_compute;

GCTensorShiftKernel::GCTensorShiftKernel()
    : _input(nullptr), _lws(gles::NDRange(1U, 1U, 1U)), _left_padding(0)
{
}

void GCTensorShiftKernel::configure(IGCTensor *input)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);

    _input = input;

    std::set<std::string> options;
    options.emplace("#define LOCAL_SIZE_X " + support::cpp11::to_string(_lws[0]));
    options.emplace("#define LOCAL_SIZE_Y " + support::cpp11::to_string(_lws[1]));
    options.emplace("#define LOCAL_SIZE_Z " + support::cpp11::to_string(_lws[2]));
    options.emplace("#define WIDTH " + support::cpp11::to_string(input->info()->dimension(0)));

    std::string dt_name = (input->info()->data_type() == DataType::F32) ? "DATA_TYPE_FP32" : "DATA_TYPE_FP16";
    options.emplace(("#define " + dt_name));

    unsigned int num_elems_written_per_iteration_x = input->info()->dimension(0) + input->info()->padding().left + input->info()->padding().right;

    std::stringstream kernel_name;
    kernel_name << "tensorshift";

    _kernel = static_cast<GCKernel>(GCKernelLibrary::get().create_kernel(kernel_name.str(), options));

    Window win;
    win.set(Window::DimX, Window::Dimension(0, num_elems_written_per_iteration_x, num_elems_written_per_iteration_x));
    win.use_tensor_dimensions(input->info()->tensor_shape(), Window::DimY);
    win.use_tensor_dimensions(input->info()->tensor_shape(), Window::DimZ);

    _left_padding = _input->info()->padding().left;

    IGCKernel::configure(win);
}

void GCTensorShiftKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    if(int(_left_padding) == 0 || !_input->needs_shifting())
    {
        return;
    }

    _kernel.use();

    // Get initial windows
    Window slice = window.first_slice_window_3D();
    slice.shift(Window::DimX, -(_input->info()->padding()).left);

    do
    {
        unsigned int idx = 0;

        add_3D_tensor_argument(idx, _input, 1, slice);

        _kernel.set_argument(idx++, static_cast<unsigned int>(_left_padding));

        _kernel.update_shader_params();
        enqueue(*this, slice, _lws);
    }
    while(window.slide_window_slice_3D(slice));
}
