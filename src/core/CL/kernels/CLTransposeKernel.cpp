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

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"

#include <set>
#include <sstream>
#include <string>

using namespace arm_compute;

void CLTransposeKernel::configure(const ICLTensor *input, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S8, DataType::QS8, DataType::QASYMM8, DataType::U16, DataType::S16, DataType::QS16, DataType::U32, DataType::S32,
                                                  DataType::F16,
                                                  DataType::F32);
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);

    TensorShape  output_shape{ input->info()->tensor_shape() };
    const size_t w_out = input->info()->dimension(1);
    const size_t h_out = input->info()->dimension(0);
    output_shape.set(0, w_out);
    output_shape.set(1, h_out);

    // Output tensor auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type(), input->info()->fixed_point_position(), input->info()->quantization_info());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(output->info()->tensor_shape(), output_shape);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);

    _input    = input;
    _output   = output;
    _lws_hint = cl::NDRange(2, 8);

    std::set<std::string> build_opts;
    std::ostringstream    data_type_in_bytes;
    data_type_in_bytes << input->info()->element_size();
    build_opts.emplace("-DDATA_TYPE_IN_BYTES=" + data_type_in_bytes.str());

    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("transpose", build_opts));

    // Configure kernel window
    const unsigned int num_elems_processed_per_iteration = max_cl_vector_width / input->info()->element_size();

    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration, num_elems_processed_per_iteration));

    AccessWindowRectangle input_access(input->info(), 0, 0, num_elems_processed_per_iteration, num_elems_processed_per_iteration);
    // TODO (COMPMID-708): Replace AccessWindowStatic with AccessWindowTranspose
    AccessWindowStatic output_access(output->info(), 0, 0, ceil_to_multiple(output->info()->dimension(0), num_elems_processed_per_iteration), ceil_to_multiple(output->info()->dimension(1),
                                     num_elems_processed_per_iteration));

    update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, input->info()->valid_region());

    ICLKernel::configure(win);
}
