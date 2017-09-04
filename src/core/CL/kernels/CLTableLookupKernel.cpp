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
#include "arm_compute/core/CL/kernels/CLTableLookupKernel.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLLut.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

#include <cstdint>
#include <string>

using namespace arm_compute;

void CLTableLookupKernel::configure(const ICLTensor *input, const ICLLut *lut, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S16);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8, DataType::S16);
    ARM_COMPUTE_ERROR_ON(lut == nullptr);
    ARM_COMPUTE_ERROR_ON(DataType::U8 != lut->type() && DataType::S16 != lut->type());
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

    // Create kernel
    std::string kernel_name = (DataType::S16 == lut->type()) ? "tablelookup_S16" : "tablelookup_U8";
    _kernel                 = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name));

    // Set lut argument
    unsigned int idx = 2 * num_arguments_per_2D_tensor(); //Skip the input and output parameters
    _kernel.setArg(idx++, lut->cl_buffer());
    if(DataType::S16 == lut->type())
    {
        _kernel.setArg(idx++, lut->index_offset());
        _kernel.setArg(idx++, static_cast<uint32_t>(lut->num_elements()));
    }

    // Configure kernel
    constexpr unsigned int num_elems_processed_per_iteration = 8;
    ICLSimple2DKernel::configure(input, output, num_elems_processed_per_iteration);
}
