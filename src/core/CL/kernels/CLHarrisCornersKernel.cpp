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
#include "arm_compute/core/CL/kernels/CLHarrisCornersKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <set>
#include <sstream>
#include <string>

using namespace arm_compute;

CLHarrisScoreKernel::CLHarrisScoreKernel()
    : _input1(nullptr), _input2(nullptr), _output(nullptr), _sensitivity(), _strength_thresh(), _norm_factor(), _border_size(0)
{
}

BorderSize CLHarrisScoreKernel::border_size() const
{
    return _border_size;
}

void CLHarrisScoreKernel::configure(const ICLImage *input1, const ICLImage *input2, ICLImage *output,
                                    int32_t block_size, float norm_factor, float strength_thresh, float sensitivity,
                                    bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input1);
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input2);
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(output);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input1, 1, DataType::S16, DataType::S32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input2, 1, DataType::S16, DataType::S32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input1, input2);
    ARM_COMPUTE_ERROR_ON(!(block_size == 3 || block_size == 5 || block_size == 7));
    ARM_COMPUTE_ERROR_ON(0.0f == norm_factor);

    _input1          = input1;
    _input2          = input2;
    _output          = output;
    _sensitivity     = sensitivity;
    _strength_thresh = strength_thresh;
    _norm_factor     = norm_factor;
    _border_size     = BorderSize(block_size / 2);

    // Select kernel
    std::stringstream harris_score_kernel_name;
    harris_score_kernel_name << "harris_score_" << block_size << "x" << block_size;

    // Create build options
    std::set<std::string> build_opts = { ("-DDATA_TYPE=" + get_cl_type_from_data_type(input1->info()->data_type())) };

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(harris_score_kernel_name.str(), build_opts));

    // Set static kernel arguments
    unsigned int idx = 3 * num_arguments_per_2D_tensor(); //Skip the input and output parameters
    _kernel.setArg(idx++, sensitivity);
    _kernel.setArg(idx++, strength_thresh);
    _kernel.setArg(idx++, norm_factor);

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 4;
    constexpr unsigned int num_elems_written_per_iteration   = 4;
    const unsigned int     num_elems_read_per_iteration      = block_size == 7 ? 10 : 8;
    const unsigned int     num_rows_read_per_iteration       = block_size;

    Window win = calculate_max_window(*_input1->info(), Steps(num_elems_processed_per_iteration), border_undefined, border_size());

    AccessWindowRectangle  input1_access(input1->info(), -border_size().left, -border_size().top, num_elems_read_per_iteration, num_rows_read_per_iteration);
    AccessWindowRectangle  input2_access(input2->info(), -border_size().left, -border_size().top, num_elems_read_per_iteration, num_rows_read_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration);

    update_window_and_padding(win, input1_access, input2_access, output_access);

    ValidRegion valid_region = intersect_valid_regions(input1->info()->valid_region(), input2->info()->valid_region());
    output_access.set_valid_region(win, valid_region, border_undefined, border_size());

    ICLKernel::configure_internal(win);

    // Set config_id for enabling LWS tuning
    _config_id = harris_score_kernel_name.str();
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(input1->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input1->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input1->info()->dimension(1));
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(input2->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input2->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input2->info()->dimension(1));
}

void CLHarrisScoreKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice = window.first_slice_window_2D();
    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input1, slice);
        add_2D_tensor_argument(idx, _input2, slice);
        add_2D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(window.slide_window_slice_2D(slice));
}
