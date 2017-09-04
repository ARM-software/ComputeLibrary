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
#include "arm_compute/core/CL/kernels/CLScaleKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"

#include <set>
#include <string>

using namespace arm_compute;

BorderSize CLScaleKernel::border_size() const
{
    return BorderSize(1);
}

void CLScaleKernel::configure(const ICLTensor *input, ICLTensor *output, InterpolationPolicy policy, bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S16);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8, DataType::S16);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

    _input  = input;
    _output = output;

    /* Compute the ratio between source width/height and destination width/height */
    const auto wr = static_cast<float>(input->info()->dimension(0)) / static_cast<float>(output->info()->dimension(0));
    const auto hr = static_cast<float>(input->info()->dimension(1)) / static_cast<float>(output->info()->dimension(1));

    /* Area interpolation behaves as Nearest Neighbour in case of up-sampling */
    if(policy == InterpolationPolicy::AREA && wr <= 1.f && hr <= 1.f)
    {
        policy = InterpolationPolicy::NEAREST_NEIGHBOR;
    }
    else
    {
        ARM_COMPUTE_ERROR_ON(policy == InterpolationPolicy::AREA);
    }

    // Create kernel
    std::set<std::string> build_opts         = { ("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type())) };
    std::string           interpolation_name = string_from_interpolation_policy(policy);
    std::transform(interpolation_name.begin(), interpolation_name.end(), interpolation_name.begin(), ::tolower);
    std::string kernel_name = "scale_" + interpolation_name;
    _kernel                 = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts));

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 4;
    const int              border_offset                     = (border_undefined) ? 0 : border_size().left;

    Window win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowStatic input_access(input->info(), -border_offset, -border_offset,
                                    input->info()->dimension(0) + border_offset, input->info()->dimension(1) + border_offset);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));

    ICLKernel::configure(win);

    // Set static kernel arguments
    unsigned int idx = 2 * num_arguments_per_2D_tensor(); //Skip the input and output parameters
    _kernel.setArg<float>(idx++, input->info()->dimension(0));
    _kernel.setArg<float>(idx++, input->info()->dimension(1));
    _kernel.setArg<float>(idx++, output->info()->dimension(0));
    _kernel.setArg<float>(idx++, output->info()->dimension(1));
}
