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
#include "arm_compute/core/CL/kernels/CLNonLinearFilterKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <set>
#include <sstream>
#include <string>

using namespace arm_compute;

CLNonLinearFilterKernel::CLNonLinearFilterKernel()
    : _border_size(0)
{
}

BorderSize CLNonLinearFilterKernel::border_size() const
{
    return _border_size;
}

void CLNonLinearFilterKernel::configure(const ICLTensor *input, ICLTensor *output, NonLinearFilterFunction function,
                                        unsigned int mask_size, MatrixPattern pattern, const uint8_t *mask,
                                        bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(mask_size != 3 && mask_size != 5);
    ARM_COMPUTE_ERROR_ON_MSG(pattern == MatrixPattern::OTHER, "MatrixPattern::OTHER is not supported!");
    ARM_COMPUTE_UNUSED(mask);

    _input       = input;
    _output      = output;
    _border_size = BorderSize(mask_size / 2);

    // Define build options
    std::set<std::string> build_opts;
    build_opts.emplace("-D" + string_from_non_linear_filter_function(function));

    // Define kernel
    std::string pattern_name = string_from_matrix_pattern(pattern);
    std::transform(pattern_name.begin(), pattern_name.end(), pattern_name.begin(), ::tolower);
    std::stringstream ss;
    ss << "non_linear_filter_" << pattern_name << mask_size << "x" << mask_size;

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(ss.str(), build_opts));

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 8;
    constexpr unsigned int num_elems_read_per_iteration      = 16;
    const unsigned int     num_rows_read_per_iteration       = mask_size;

    Window win = calculate_max_window(*input->info(), num_elems_processed_per_iteration, border_undefined, border_size());

    AccessWindowRectangle  input_access(input->info(), -border_size().left, -border_size().top, num_elems_read_per_iteration, num_rows_read_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());

    ICLKernel::configure(win);
}
