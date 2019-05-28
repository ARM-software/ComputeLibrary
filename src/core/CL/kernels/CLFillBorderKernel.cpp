/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLFillBorderKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <cstdint>
#include <set>
#include <sstream>
#include <string>

using namespace arm_compute;

CLFillBorderKernel::CLFillBorderKernel()
    : ICLKernel(), _tensor(nullptr)
{
}

bool CLFillBorderKernel::is_parallelisable() const
{
    return false;
}

template <class T>
void CLFillBorderKernel::set_constant_border(unsigned int idx, const PixelValue &constant_border_value)
{
    T value;
    constant_border_value.get(value);
    ICLKernel::add_argument<T>(idx, static_cast<T>(value));
}

void CLFillBorderKernel::configure(ICLTensor *tensor, BorderSize border_size, BorderMode border_mode, const PixelValue &constant_border_value)
{
    ARM_COMPUTE_ERROR_ON(tensor == nullptr);
    ARM_COMPUTE_ERROR_ON(tensor->info()->num_channels() != 1);

    border_size.limit(tensor->info()->padding());

    // If there is no border: early exit
    if(border_size.empty() || border_mode == BorderMode::UNDEFINED)
    {
        return;
    }

    // Select appropriate kernel
    std::string kernel_name = "fill_image_borders_" + lower_string(string_from_border_mode(border_mode));

    const DataType dt = tensor->info()->data_type();

    // Define build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_underlying_cl_type_from_data_type(dt));
    build_opts.add_option("-DBORDER_SIZE_TOP=" + support::cpp11::to_string(border_size.top));
    build_opts.add_option("-DBORDER_SIZE_BOTTOM=" + support::cpp11::to_string(border_size.bottom));
    build_opts.add_option("-DBORDER_SIZE_LEFT=" + support::cpp11::to_string(border_size.left));
    build_opts.add_option("-DBORDER_SIZE_RIGHT=" + support::cpp11::to_string(border_size.right));

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));
    _tensor = tensor;

    // Create static kernel arguments
    const unsigned int valid_width  = tensor->info()->valid_region().shape[0];
    const unsigned int valid_height = tensor->info()->valid_region().shape[1];
    const cl_int2      valid_region_coords =
    {
        {
            static_cast<cl_int>(tensor->info()->valid_region().anchor[0]),
            static_cast<cl_int>(tensor->info()->valid_region().anchor[1]),
        }
    };
    const unsigned int total_valid_width = border_size.left + valid_width + border_size.right;

    // Set static kernel arguments
    unsigned int idx = num_arguments_per_3D_tensor(); //Skip the tensor parameters
    ICLKernel::add_argument<cl_uint>(idx, valid_width);
    ICLKernel::add_argument<cl_uint>(idx, valid_height);
    ICLKernel::add_argument<cl_int2>(idx, valid_region_coords);
    if(BorderMode::CONSTANT == border_mode)
    {
        switch(dt)
        {
            case DataType::U8:
            case DataType::QASYMM8:
                set_constant_border<uint8_t>(idx, constant_border_value);
                break;
            case DataType::S8:
                set_constant_border<int8_t>(idx, constant_border_value);
                break;
            case DataType::U16:
                set_constant_border<uint16_t>(idx, constant_border_value);
                break;
            case DataType::S16:
                set_constant_border<int16_t>(idx, constant_border_value);
                break;
            case DataType::U32:
                set_constant_border<uint32_t>(idx, constant_border_value);
                break;
            case DataType::S32:
                set_constant_border<int32_t>(idx, constant_border_value);
                break;
            case DataType::F32:
                static_assert(sizeof(float) == 4, "Float must be 32 bit");
                set_constant_border<float>(idx, constant_border_value);
                break;
            case DataType::F16:
                static_assert(sizeof(cl_half) == sizeof(half), "Half must be same size as cl_half");
                static_assert(sizeof(cl_half) == 2, "Half must be 16 bit");
                set_constant_border<half>(idx, constant_border_value);
                break;
            default:
                ARM_COMPUTE_ERROR("Not handled");
        }
    }

    // Configure kernel window
    Window win;
    win.set(Window::DimX, Window::Dimension(0, total_valid_width + valid_height));
    win.set(Window::DimY, Window::Dimension(0, 1, 1));
    win.use_tensor_dimensions(tensor->info()->tensor_shape(), Window::DimZ);
    ICLKernel::configure_internal(win);

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(dt));
    _config_id += "_";
    _config_id += support::cpp11::to_string(tensor->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(tensor->info()->dimension(1));
    _config_id += "_";
    _config_id += lower_string(string_from_border_mode(border_mode));
}

void CLFillBorderKernel::run(const Window &window, cl::CommandQueue &queue)
{
    // Border mode undefined or border width == 0
    if(_kernel() == nullptr)
    {
        return;
    }

    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

    Window collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice     = collapsed.first_slice_window_3D();

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _tensor, slice);
        enqueue(queue, *this, slice, cl::NullRange);
    }
    while(collapsed.slide_window_slice_3D(slice));
}
