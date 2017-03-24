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
#include "arm_compute/core/CL/kernels/CLTransposeKernel.h"

#include "arm_compute/core/AccessWindowTranspose.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"

#include <set>
#include <sstream>
#include <string>

using namespace arm_compute;

void CLTransposeKernel::configure(const ICLTensor *input, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S8, DataType::U16, DataType::S16, DataType::U32, DataType::S32, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8, DataType::S8, DataType::U16, DataType::S16, DataType::U32, DataType::S32, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(0) != output->info()->dimension(1));
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(1) != output->info()->dimension(0));

    _input    = input;
    _output   = output;
    _lws_hint = cl::NDRange(2, 8);

    std::set<std::string> build_opts;
    std::ostringstream    data_type_in_bytes;
    data_type_in_bytes << input->info()->element_size();
    build_opts.emplace("-DDATA_TYPE_IN_BYTES=" + data_type_in_bytes.str());

    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("transpose", build_opts));

    // Configure kernel window
    const unsigned int processed_elements = max_cl_vector_width / input->info()->element_size();

    Window                win = calculate_max_window(*input->info(), Steps(processed_elements, processed_elements));
    AccessWindowTranspose output_access(output->info(), 0, 0, processed_elements, processed_elements);

    update_window_and_padding(win,
                              AccessWindowRectangle(input->info(), 0, 0, processed_elements, processed_elements),
                              output_access);

    output_access.set_valid_region(win, input->info()->valid_region());

    ICLKernel::configure(win);
}
