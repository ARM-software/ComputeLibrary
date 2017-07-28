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
#include "arm_compute/core/CL/kernels/CLWarpPerspectiveKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"

#include <cstddef>
#include <set>
#include <sstream>
#include <string>

using namespace arm_compute;

namespace
{
inline void options_add_matrix(std::set<std::string> &options, const float *matrix, size_t size)
{
    for(size_t i = 0; i < size; ++i)
    {
        std::stringstream mat_str;
        mat_str << "-DMAT" << i << "=" << matrix[i] << " ";
        options.insert(mat_str.str());
    }
}
} // namespace

BorderSize CLWarpPerspectiveKernel::border_size() const
{
    return BorderSize(1);
}

void CLWarpPerspectiveKernel::configure(const ICLTensor *input, ICLTensor *output, const float *matrix, InterpolationPolicy policy)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(InterpolationPolicy::AREA == policy);

    _input  = input;
    _output = output;

    // Create build options
    std::set<std::string> options;
    options_add_matrix(options, matrix, 9);
    options.emplace(("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type())));

    // Create kernel
    std::string interpolation_name = string_from_interpolation_policy(policy);
    std::transform(interpolation_name.begin(), interpolation_name.end(), interpolation_name.begin(), ::tolower);
    std::string kernel_name = "warp_perspective_" + interpolation_name;
    _kernel                 = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, options));

    // Set static kernel arguments
    unsigned int idx = 2 * num_arguments_per_2D_tensor(); //Skip the input and output parameters
    _kernel.setArg<cl_int>(idx++, input->info()->dimension(0));
    _kernel.setArg<cl_int>(idx++, input->info()->dimension(1));

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 4;

    Window win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowStatic     input_access(input->info(), -border_size().left, -border_size().top, input->info()->dimension(0) + border_size().right, input->info()->dimension(1) + border_size().bottom);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));

    ICLKernel::configure(win);
}
