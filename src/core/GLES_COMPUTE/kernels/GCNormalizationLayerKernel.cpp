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
#include "arm_compute/core/GLES_COMPUTE/kernels/GCNormalizationLayerKernel.h"

#include "arm_compute/core/GLES_COMPUTE/GCHelpers.h"
#include "arm_compute/core/GLES_COMPUTE/GCKernelLibrary.h"
#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <string>

using namespace arm_compute;

GCNormalizationLayerKernel::GCNormalizationLayerKernel()
    : _input(nullptr), _squared_input(nullptr), _output(nullptr), _border_size(0)
{
}

BorderSize GCNormalizationLayerKernel::border_size() const
{
    return _border_size;
}

void GCNormalizationLayerKernel::configure(const IGCTensor *input, const IGCTensor *squared_input, IGCTensor *output, NormalizationLayerInfo norm_info)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MSG(!(norm_info.norm_size() % 2), "Normalization size should be odd");
    ARM_COMPUTE_ERROR_ON_MSG(norm_info.type() == NormType::IN_MAP_2D, "2D In-Map Normalization not implemented");

    // Set build options
    std::set<std::string> build_opts;

    _input         = input;
    _squared_input = squared_input;
    _output        = output;

    const bool         is_in_map    = norm_info.is_in_map();
    const unsigned int border_width = is_in_map ? std::min(norm_info.norm_size() / 2, 3U) : 0;
    _border_size                    = BorderSize(0, border_width);

    // Set kernel static arguments
    std::string func_name = ((norm_info.type() == NormType::IN_MAP_1D) ? "IN_MAP_1D" : "CROSS_MAP");
    build_opts.emplace(("#define " + func_name));
    build_opts.emplace(("#define COEFF " + float_to_string_with_full_precision(norm_info.scale_coeff())));
    build_opts.emplace(("#define BETA " + float_to_string_with_full_precision(norm_info.beta())));
    build_opts.emplace(("#define KAPPA " + float_to_string_with_full_precision(norm_info.kappa())));
    build_opts.emplace(("#define RADIUS " + support::cpp11::to_string(norm_info.norm_size() / 2)));
    build_opts.emplace(("#define LOCAL_SIZE_X " + support::cpp11::to_string(1)));
    build_opts.emplace(("#define LOCAL_SIZE_Y " + support::cpp11::to_string(1)));
    build_opts.emplace(("#define LOCAL_SIZE_Z " + support::cpp11::to_string(1)));

    // Create kernel
    _kernel = static_cast<GCKernel>(GCKernelLibrary::get().create_kernel("normalization_layer", build_opts));

    // Configure kernel window
    const unsigned int num_elems_processed_per_iteration = 1;
    const unsigned int num_elems_read_per_iteration      = num_elems_processed_per_iteration + 2 * (norm_info.norm_size() / 2);

    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(input->info(), -_border_size.left, num_elems_read_per_iteration);
    AccessWindowHorizontal squared_input_access(squared_input->info(), -_border_size.left, num_elems_read_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win, input_access, squared_input_access, output_access);

    output_access.set_valid_region(win, input->info()->valid_region());

    IGCKernel::configure(win);
}

void GCNormalizationLayerKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    _kernel.use();

    Window slice = window.first_slice_window_3D();

    do
    {
        unsigned int idx     = 0;
        unsigned int binding = 1;
        add_3D_tensor_argument(idx, _input, binding++, slice);
        add_3D_tensor_argument(idx, _squared_input, binding++, slice);
        add_3D_tensor_argument(idx, _output, binding++, slice);

        _kernel.update_shader_params();

        enqueue(*this, slice);
    }
    while(window.slide_window_slice_3D(slice));
}
