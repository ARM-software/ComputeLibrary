/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "src/core/AccessWindowStatic.h"
#include "src/core/AccessWindowTranspose.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <set>
#include <sstream>
#include <string>

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

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    // Configure kernel window
    const unsigned int num_elems_processed_per_iteration = max_cl_vector_width / input->element_size();

    Window win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration, num_elems_processed_per_iteration));

    AccessWindowRectangle input_access(input, 0, 0, num_elems_processed_per_iteration, num_elems_processed_per_iteration);

    bool window_changed = update_window_and_padding(win, input_access);

    if(output->total_size() != 0)
    {
        AccessWindowTranspose output_access(output, 0, 0, num_elems_processed_per_iteration, num_elems_processed_per_iteration);

        window_changed = window_changed || update_window_and_padding(win, output_access);

        output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

Status CLTransposeKernel::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get()).first);
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

    _input  = input;
    _output = output;

    std::set<std::string> build_opts;
    std::ostringstream    data_type_in_bytes;
    data_type_in_bytes << input->info()->element_size();
    build_opts.emplace("-DDATA_TYPE_IN_BYTES=" + data_type_in_bytes.str());

    _kernel = create_kernel(compile_context, "transpose", build_opts);

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second, cl::NDRange(2, 8));
}
} // namespace arm_compute