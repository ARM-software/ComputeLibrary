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
#include "arm_compute/core/CL/kernels/CLThresholdKernel.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <string>

using namespace arm_compute;

void CLThresholdKernel::configure(const ICLTensor *input, ICLTensor *output, uint8_t threshold,
                                  uint8_t false_value, uint8_t true_value, ThresholdType type, uint8_t upper)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);

    // Construct kernel name
    std::string kernel_name = "threshold";

    switch(type)
    {
        case ThresholdType::BINARY:
            kernel_name += "_binary";
            break;
        case ThresholdType::RANGE:
            kernel_name += "_range";
            break;
        default:
            ARM_COMPUTE_ERROR("Thresholding type not recognized");
            break;
    }

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name));

    // Set arguments
    unsigned int idx = 2 * num_arguments_per_2D_tensor(); //Skip the input and output parameters
    _kernel.setArg(idx++, false_value);
    _kernel.setArg(idx++, true_value);
    _kernel.setArg(idx++, threshold);

    if(ThresholdType::RANGE == type)
    {
        _kernel.setArg(idx++, upper);
    }

    // Make sure _kernel is initialized before calling the parent's configure
    constexpr unsigned int num_elems_processed_per_iteration = 16;
    ICLSimple2DKernel::configure(input, output, num_elems_processed_per_iteration);
}
