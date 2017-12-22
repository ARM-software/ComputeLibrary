/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/core/GLES_COMPUTE/kernels/GCFillBorderKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/GLES_COMPUTE/GCHelpers.h"
#include "arm_compute/core/GLES_COMPUTE/GCKernelLibrary.h"
#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"
#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <cstdint>
#include <set>
#include <string>

using namespace arm_compute;

GCFillBorderKernel::GCFillBorderKernel()
    : IGCKernel(), _tensor(nullptr)
{
}

bool GCFillBorderKernel::is_parallelisable() const
{
    return false;
}

template <class T>
void GCFillBorderKernel::set_constant_border(unsigned int idx, const PixelValue &constant_border_value)
{
    T value;
    constant_border_value.get(value);
    _kernel.set_argument(idx, static_cast<T>(value));
}

void GCFillBorderKernel::configure(const IGCTensor *tensor, BorderSize border_size, BorderMode border_mode, const PixelValue &constant_border_value)
{
    ARM_COMPUTE_ERROR_ON(tensor == nullptr);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(tensor, 1, DataType::F32, DataType::F16);
    ARM_COMPUTE_ERROR_ON(tensor->info()->num_channels() != 1);

    border_size.limit(tensor->info()->padding());

    // If there is no border: early exit
    if(border_size.empty() || border_mode == BorderMode::UNDEFINED)
    {
        return;
    }

    // Select appropriate kernel
    std::string kernel_name = "fill_image_borders_" + lower_string(string_from_border_mode(border_mode));

    // Define build options
    std::set<std::string> build_opts;
    build_opts.emplace("#define LOCAL_SIZE_X " + support::cpp11::to_string(1));
    build_opts.emplace("#define LOCAL_SIZE_Y " + support::cpp11::to_string(1));
    build_opts.emplace("#define LOCAL_SIZE_Z " + support::cpp11::to_string(1));
    build_opts.emplace("#define BORDER_SIZE_TOP " + support::cpp11::to_string(border_size.top));
    build_opts.emplace("#define BORDER_SIZE_BOTTOM " + support::cpp11::to_string(border_size.bottom));
    build_opts.emplace("#define BORDER_SIZE_LEFT " + support::cpp11::to_string(border_size.left));
    build_opts.emplace("#define BORDER_SIZE_RIGHT " + support::cpp11::to_string(border_size.right));

    if(border_mode == BorderMode::REPLICATE)
    {
        build_opts.emplace("#define FILL_IMAGE_BORDERS_REPLICATE\n");
    }
    else
    {
        build_opts.emplace("#define FILL_IMAGE_BORDERS_CONSTANT\n");
    }

    switch(tensor->info()->data_type())
    {
        case DataType::F16:
            build_opts.emplace("#define DATA_TYPE_FP16");
            break;

        case DataType::F32:
            build_opts.emplace("#define DATA_TYPE_FP32");
            break;

        default:
            ARM_COMPUTE_ERROR("Current data type is not supported");
            break;
    }

    // Create kernel
    _kernel = static_cast<GCKernel>(GCKernelLibrary::get().create_kernel(kernel_name, build_opts));
    _tensor = tensor;

    // Create static kernel arguments
    const unsigned int valid_width       = tensor->info()->valid_region().shape[0];
    const unsigned int valid_height      = tensor->info()->valid_region().shape[1];
    const unsigned int total_valid_width = border_size.left + valid_width + border_size.right;

    // Set static kernel arguments
    unsigned int idx = num_arguments_per_3D_tensor(); //Skip the tensor parameters
    _kernel.set_argument(idx++, valid_width);
    _kernel.set_argument(idx++, valid_height);
    _kernel.set_argument(idx++, tensor->info()->valid_region().anchor[0]);
    _kernel.set_argument(idx++, tensor->info()->valid_region().anchor[1]);

    if(BorderMode::CONSTANT == border_mode)
    {
        set_constant_border<float>(idx++, constant_border_value);
    }

    // Configure kernel window
    Window win;
    win.set(Window::DimX, Window::Dimension(0, total_valid_width + valid_height));
    win.set(Window::DimY, Window::Dimension(0, 1, 1));
    win.use_tensor_dimensions(tensor->info()->tensor_shape(), Window::DimZ);

    IGCKernel::configure(win);
}

void GCFillBorderKernel::run(const Window &window)
{
    // Border mode undefined or border width == 0
    if(_kernel.get_program() == 0)
    {
        return;
    }

    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(IGCKernel::window(), window);

    _kernel.use();
    Window slice = window.first_slice_window_3D();

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _tensor, 1, slice);

        _kernel.update_shader_params();

        enqueue(*this, slice);
    }
    while(window.slide_window_slice_3D(slice));
}
