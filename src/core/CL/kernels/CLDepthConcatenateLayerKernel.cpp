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
#include "arm_compute/core/CL/kernels/CLDepthConcatenateLayerKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "support/ToolchainSupport.h"

#include <map>

using namespace arm_compute;

CLDepthConcatenateLayerKernel::CLDepthConcatenateLayerKernel()
    : _input(nullptr), _output(nullptr), _top_bottom(0), _left_right(0), _depth_offset(0)
{
}

BorderSize CLDepthConcatenateLayerKernel::border_size() const
{
    return BorderSize(_top_bottom, _left_right);
}

void CLDepthConcatenateLayerKernel::configure(const ICLTensor *input, unsigned int depth_offset, ICLTensor *output)
{
    static std::map<int, std::pair<std::string, int>> configs_map =
    {
        { 1, { "uchar", 16 } },
        { 2, { "ushort", 8 } },
        { 4, { "uint", 4 } },
        { 8, { "ulong", 2 } },
    };

    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT_POSITION(input, output);
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(2) + depth_offset > output->info()->dimension(2));
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(0) > output->info()->dimension(0));
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(1) > output->info()->dimension(1));
    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(3, input, output);
    ARM_COMPUTE_ERROR_ON(configs_map.find(input->info()->element_size()) == configs_map.end());

    // The gaps between the two lowest dimensions of input and output need to be divisible by 2
    // Otherwise it is not clear how the padding should be added onto the input tensor
    ARM_COMPUTE_ERROR_ON((output->info()->dimension(0) - input->info()->dimension(0)) % 2);
    ARM_COMPUTE_ERROR_ON((output->info()->dimension(1) - input->info()->dimension(1)) % 2);

    _input        = input;
    _output       = output;
    _depth_offset = depth_offset;

    // Add build options
    auto                  config = configs_map.find(static_cast<int>(input->info()->element_size()));
    std::set<std::string> build_opts;
    build_opts.emplace(("-DDATA_TYPE=" + config->second.first));
    build_opts.emplace(("-DVEC_SIZE=" + support::cpp11::to_string(config->second.second)));

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("concatenate_depth", build_opts));

    // Configure kernel window
    _left_right = (output->info()->dimension(0) - input->info()->dimension(0)) / 2;
    _top_bottom = (output->info()->dimension(1) - input->info()->dimension(1)) / 2;

    const unsigned int num_elems_processed_per_iteration = 16 / input->info()->element_size();
    const unsigned int num_elems_read_per_iteration      = 16 / input->info()->element_size();
    const unsigned int num_rows_read_per_iteration       = 1;

    // The window needs to be based on input as we copy all the depths of input
    Window win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration));
    win.set(Window::DimZ, Window::Dimension(0, input->info()->tensor_shape().z(), 1));

    AccessWindowRectangle  input_access(input->info(), -_left_right, -_top_bottom, num_elems_read_per_iteration, num_rows_read_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);
    update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, ValidRegion(Coordinates(0, 0), output->info()->tensor_shape()));

    ICLKernel::configure(win);
}

void CLDepthConcatenateLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice = window.first_slice_window_3D();

    const int offset_to_first_elements_in_bytes = _depth_offset * _output->info()->strides_in_bytes()[2];

    unsigned int  idx = 2 * num_arguments_per_3D_tensor(); // Skip the input and output parameters
    const cl_int3 offsets =
    {
        {
            static_cast<cl_int>(_left_right),
            static_cast<cl_int>(_top_bottom),
            static_cast<cl_int>(offset_to_first_elements_in_bytes),
        }
    };
    _kernel.setArg<cl_int3>(idx, offsets);

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice);
    }
    while(window.slide_window_slice_3D(slice));
}
