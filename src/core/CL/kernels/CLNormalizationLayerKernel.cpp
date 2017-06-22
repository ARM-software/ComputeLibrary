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
#include "arm_compute/core/CL/kernels/CLNormalizationLayerKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/FixedPoint.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

using namespace arm_compute;

CLNormalizationLayerKernel::CLNormalizationLayerKernel()
    : _input(nullptr), _squared_input(nullptr), _output(nullptr), _border_size(0), _is_in_map(false)
{
}

BorderSize CLNormalizationLayerKernel::border_size() const
{
    return _border_size;
}

void CLNormalizationLayerKernel::configure(const ICLTensor *input, const ICLTensor *squared_input, ICLTensor *output, NormalizationLayerInfo norm_info)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);

    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->tensor_shape(), 1, input->info()->data_type(), input->info()->fixed_point_position());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, squared_input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, squared_input, output);
    ARM_COMPUTE_ERROR_ON_MSG(!(norm_info.norm_size() % 2), "Normalization size should be odd");
    ARM_COMPUTE_ERROR_ON_MSG(norm_info.type() == NormType::IN_MAP_2D, "2D In-Map Normalization not implemented");
    if(is_data_type_fixed_point(input->info()->data_type()))
    {
        ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, squared_input, output);
        ARM_COMPUTE_ERROR_ON_VALUE_NOT_REPRESENTABLE_IN_FIXED_POINT(norm_info.beta(), input);
        ARM_COMPUTE_ERROR_ON_VALUE_NOT_REPRESENTABLE_IN_FIXED_POINT(norm_info.kappa(), input);
        ARM_COMPUTE_ERROR_ON_VALUE_NOT_REPRESENTABLE_IN_FIXED_POINT(norm_info.scale_coeff(), input);
    }

    _input         = input;
    _squared_input = squared_input;
    _output        = output;

    _is_in_map                      = (norm_info.type() != NormType::CROSS_MAP);
    const unsigned int border_width = _is_in_map ? std::min(norm_info.norm_size() / 2, 3U) : 0;
    _border_size                    = BorderSize(0, border_width);

    const unsigned int num_elems_processed_per_iteration = (is_data_type_fixed_point(input->info()->data_type())) ? 16 : 4;
    const unsigned int num_elems_read_per_iteration      = num_elems_processed_per_iteration + 2 * (norm_info.norm_size() / 2);

    // Set build options
    std::set<std::string> build_opts;
    build_opts.emplace(("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type())));
    if(is_data_type_fixed_point(input->info()->data_type()))
    {
        build_opts.emplace(("-DFIXED_POINT_POSITION=" + support::cpp11::to_string(input->info()->fixed_point_position())));
    }
    build_opts.emplace(("-DCOEFF=" + support::cpp11::to_string(norm_info.scale_coeff())));
    build_opts.emplace(("-DBETA=" + support::cpp11::to_string(norm_info.beta())));
    build_opts.emplace(("-DKAPPA=" + support::cpp11::to_string(norm_info.kappa())));
    build_opts.emplace(("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration)));

    // Create kernel
    std::string kernel_name = (norm_info.type() == NormType::IN_MAP_1D) ? "normalization_layer_in_map_1D" : "normalization_layer_cross_map";
    _kernel                 = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts));

    // Set kernel static arguments
    unsigned int idx = 3 * num_arguments_per_3D_tensor(); // Skip the input and output parameters
    _kernel.setArg<cl_uint>(idx++, norm_info.norm_size() / 2);

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(input->info(), -_border_size.left, num_elems_read_per_iteration);
    AccessWindowHorizontal squared_input_access(squared_input->info(), -_border_size.left, num_elems_read_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win, input_access, squared_input_access, output_access);

    output_access.set_valid_region(win, input->info()->valid_region());

    ICLKernel::configure(win);
}

void CLNormalizationLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    const int collapsed_dimension = _is_in_map ? Window::DimZ : 4;
    Window    window_collapsed    = window.collapse_if_possible(ICLKernel::window(), collapsed_dimension);
    Window    slice               = window_collapsed.first_slice_window_3D();

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _squared_input, slice);
        add_3D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice);
    }
    while(window_collapsed.slide_window_slice_3D(slice));
}
