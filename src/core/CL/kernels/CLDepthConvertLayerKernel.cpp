/*
 * Copyright (c) 2016-2018 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLDepthConvertLayerKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"

#include <cstddef>
#include <set>
#include <string>

using namespace arm_compute;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, ConvertPolicy policy, uint32_t shift)
{
    ARM_COMPUTE_UNUSED(policy);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON(input == output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S16,
                                                         DataType::U16, DataType::U32, DataType::S32,
                                                         DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8, DataType::S16,
                                                         DataType::U16, DataType::U32, DataType::S32,
                                                         DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == output->data_type(), "Input and output data types must be different");
    ARM_COMPUTE_RETURN_ERROR_ON(shift >= 8);

    // Check if convertion is supported
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::U8 && (output->data_type() != DataType::U16 && output->data_type() != DataType::S16
                                                                           && output->data_type() != DataType::U32 && output->data_type() != DataType::S32),
                                    "Only data types supported [in] U8 -> [out] U16, S16, U32, S32");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::U16 && (output->data_type() != DataType::U8 && output->data_type() != DataType::U32
                                                                            && output->data_type() != DataType::S32),
                                    "Only data types supported [in] U16 ->  [out] U8, U32, S32");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::S16 && (output->data_type() != DataType::U8 && output->data_type() != DataType::U32
                                                                            && output->data_type() != DataType::S32),
                                    "Only data types supported [in] S16 ->  [out] U8, U32, S32");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::U32 && (output->data_type() != DataType::U8 && output->data_type() != DataType::U16
                                                                            && output->data_type() != DataType::S16),
                                    "Only data types supported [in] U32 ->  [out] U8, U16, S16");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::S32 && (output->data_type() != DataType::U8 && output->data_type() != DataType::U16
                                                                            && output->data_type() != DataType::S16),
                                    "Only data types supported [in] S32 ->  [out] U8, U16, S16");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::F16 && output->data_type() != DataType::F32,
                                    "Only data types supported [in] F16 ->  [out] F32");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::F32 && output->data_type() != DataType::F16,
                                    "Only data types supported [in] F32 ->  [out] F16");

    // Validate in case of configured output
    if(output->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
    }

    return Status{};
}
} // namespace

void CLDepthConvertLayerKernel::configure(const ICLTensor *input, ICLTensor *output, ConvertPolicy policy, uint32_t shift)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Auto initialize output shape if not initialized (We can only auto-configure the shape, datatype must be given)
    set_shape_if_empty(*output->info(), input->info()->tensor_shape());

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), policy, shift));

    // Get data sizes
    const size_t input_size  = data_size_from_type(input->info()->data_type());
    const size_t output_size = data_size_from_type(output->info()->data_type());

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE_IN=" + get_cl_type_from_data_type(input->info()->data_type()));
    build_opts.add_option("-DDATA_TYPE_OUT=" + get_cl_type_from_data_type(output->info()->data_type()));
    // Down conversions from float always SATURATE as out-of-bounds conversion from float->integer is implementation defined
    build_opts.add_option_if(input_size > output_size, ((policy == ConvertPolicy::WRAP) && !is_data_type_float(input->info()->data_type())) ? "-DWRAP" : "-DSATURATE");
    build_opts.add_option_if(is_data_type_float(input->info()->data_type()), "-DIS_DATA_TYPE_FLOAT");

    // Create kernel
    const std::string kernel_name = (input_size > output_size) ? "convert_depth_down" : "convert_depth_up";
    _kernel                       = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));

    // Set shift arg
    unsigned int idx = 2 * num_arguments_per_2D_tensor(); //Skip the input and output parameters
    _kernel.setArg(idx++, shift);

    // Configure kernel
    constexpr unsigned int num_elems_processed_per_iteration = 16;
    ICLSimple2DKernel::configure(input, output, num_elems_processed_per_iteration);
}

Status CLDepthConvertLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, ConvertPolicy policy, uint32_t shift)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, policy, shift));

    return Status{};
}
