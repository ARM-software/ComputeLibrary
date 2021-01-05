/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "src/core/CL/kernels/CLTransposeKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace
{
TensorShape transposed_tensor_shape(const TensorShape &in)
{
    TensorShape  output_shape{ in };
    const size_t w_out = in[1];
    const size_t h_out = in[0];
    output_shape.set(0, w_out);
    output_shape.set(1, h_out);

    return output_shape;
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);

    if(output->total_size() != 0)
    {
        const TensorInfo tensor_info = input->clone()->set_tensor_shape(transposed_tensor_shape(input->tensor_shape()));

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(output, &tensor_info);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    }

    return Status{};
}
} // namespace

Status CLTransposeKernel::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output));
    return Status{};
}

void CLTransposeKernel::configure(const ICLTensor *input, ICLTensor *output)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output);
}

void CLTransposeKernel::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Output tensor auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(transposed_tensor_shape(input->info()->tensor_shape())));

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info()));
    auto padding_info = get_padding_info({ input, output });

    _input  = input;
    _output = output;

    const unsigned int vec_size_x           = adjust_vec_size(max_cl_vector_width / input->info()->element_size(), input->info()->dimension(0));
    const int          vec_size_x_leftovers = input->info()->dimension(0) % vec_size_x;
    const unsigned int vec_size_y           = adjust_vec_size(max_cl_vector_width / input->info()->element_size(), input->info()->dimension(1));
    const int          vec_size_y_leftovers = input->info()->dimension(1) % vec_size_y;

    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE_IN_BYTES=" + support::cpp11::to_string(input->info()->element_size()));
    build_opts.add_option("-DVEC_SIZE_X=" + support::cpp11::to_string(vec_size_x));
    build_opts.add_option("-DVEC_SIZE_LEFTOVER_X=" + support::cpp11::to_string(vec_size_x_leftovers));
    build_opts.add_option("-DVEC_SIZE_Y=" + support::cpp11::to_string(vec_size_y));
    build_opts.add_option("-DVEC_SIZE_LEFTOVER_Y=" + support::cpp11::to_string(vec_size_y_leftovers));

    _kernel = create_kernel(compile_context, "transpose", build_opts.options());

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps(vec_size_x, vec_size_y));
    ICLKernel::configure_internal(win, cl::NDRange(2, 8));
    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}
} // namespace arm_compute