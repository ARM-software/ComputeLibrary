/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLUpsampleLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

namespace arm_compute
{
CLUpsampleLayerKernel::CLUpsampleLayerKernel()
    : _input(nullptr), _output(nullptr), _info(), _data_layout(DataLayout::UNKNOWN), _num_elems_processed_per_iteration_input_x()
{
}

Status CLUpsampleLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const Size2D &info, const InterpolationPolicy upsampling_policy)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_UNUSED(upsampling_policy);

    DataLayout data_layout = input->data_layout();
    const int  idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int  idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(output->dimension(idx_width) != info.x() * input->dimension(idx_width));
    ARM_COMPUTE_RETURN_ERROR_ON(output->dimension(idx_height) != info.y() * input->dimension(idx_height));
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(info.x() != 2 || info.y() != 2, "Only stride 2 is supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(upsampling_policy != InterpolationPolicy::NEAREST_NEIGHBOR, "Only nearest neighbor policy supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);

    return Status{};
}

void CLUpsampleLayerKernel::configure(const ICLTensor *input, ICLTensor *output, const Size2D &info, const InterpolationPolicy upsampling_policy)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_UNUSED(upsampling_policy);

    _input                                     = input;
    _output                                    = output;
    _info                                      = info;
    _data_layout                               = input->info()->data_layout();
    _num_elems_processed_per_iteration_input_x = 1;

    TensorShape output_shape = misc::shape_calculator::compute_upsample_shape(*input->info(), info);
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type());
    output->info()->set_data_layout(_data_layout);

    unsigned int num_elems_processed_per_iteration_x = 16;
    const int    output_width_x                      = output->info()->dimension(0);
    const bool   multi_access_x                      = ((output_width_x / num_elems_processed_per_iteration_x) > 0);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(CLUpsampleLayerKernel::validate(input->info(), output->info(), info, upsampling_policy));

    Window win{};

    switch(_data_layout)
    {
        case DataLayout::NCHW:
        {
            win = calculate_max_window(*output->info());
            win.set(Window::DimY, Window::Dimension(win.y().start(), win.y().end(), info.y()));
            if(multi_access_x)
            {
                _num_elems_processed_per_iteration_input_x = num_elems_processed_per_iteration_x / info.x();
                win.set(Window::DimX, Window::Dimension(win.x().start(), ceil_to_multiple(win.x().end(), num_elems_processed_per_iteration_x), num_elems_processed_per_iteration_x));
            }
            break;
        }
        case DataLayout::NHWC:
        {
            win = calculate_max_window(*output->info());
            win.set(Window::DimY, Window::Dimension(win.y().start(), win.y().end(), info.x()));
            win.set(Window::DimZ, Window::Dimension(win.z().start(), win.z().end(), info.y()));
            if(multi_access_x)
            {
                _num_elems_processed_per_iteration_input_x = num_elems_processed_per_iteration_x;
                win.set(Window::DimX, Window::Dimension(win.x().start(), ceil_to_multiple(win.x().end(),
                                                                                          num_elems_processed_per_iteration_x),
                                                        num_elems_processed_per_iteration_x));
            }
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not implemented");
    }

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option(("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type())));
    build_opts.add_option_if(multi_access_x, "-DVEC_SIZE_IN=" + support::cpp11::to_string(_num_elems_processed_per_iteration_input_x));
    build_opts.add_option_if(multi_access_x, "-DVEC_SIZE_OUT=" + support::cpp11::to_string(num_elems_processed_per_iteration_x));
    build_opts.add_option_if(multi_access_x, "-DLAST_ACCESSED_X_IN=" + support::cpp11::to_string(std::max<int>(_input->info()->dimension(0) - _num_elems_processed_per_iteration_input_x, 0)));
    build_opts.add_option_if(multi_access_x, "-DLAST_ACCESSED_X_OUT=" + support::cpp11::to_string(std::max<int>(output_width_x - num_elems_processed_per_iteration_x, 0)));
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("upsample_layer_" + lower_string(string_from_data_layout(input->info()->data_layout())), build_opts.options()));

    ICLKernel::configure_internal(win);
}

void CLUpsampleLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window collapsed_window = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice_out        = collapsed_window.first_slice_window_3D();
    Window slice_in         = collapsed_window.first_slice_window_3D();

    switch(_data_layout)
    {
        case DataLayout::NCHW:
            slice_in.set(Window::DimX, Window::Dimension(0, _input->info()->dimension(0), _num_elems_processed_per_iteration_input_x));
            slice_in.set(Window::DimY, Window::Dimension(0, _input->info()->dimension(1), 1));
            break;
        case DataLayout::NHWC:
            slice_in.set(Window::DimY, Window::Dimension(0, _input->info()->dimension(1), 1));
            slice_in.set(Window::DimZ, Window::Dimension(0, _input->info()->dimension(2), 1));
            break;
        default:
            ARM_COMPUTE_ERROR("Not implemented");
    }

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice_in);
        add_3D_tensor_argument(idx, _output, slice_out);
        enqueue(queue, *this, slice_out);
    }
    while(collapsed_window.slide_window_slice_3D(slice_out) && collapsed_window.slide_window_slice_3D(slice_in));
}
} // namespace arm_compute
