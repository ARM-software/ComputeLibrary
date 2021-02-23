/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "src/core/CL/kernels/CLRangeKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Utils.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace
{
constexpr unsigned int vector_size_byte_opencl = 16;

Status validate_arguments(const ITensorInfo *output, const float start, const float end, const float step)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output,
                                                         1,
                                                         DataType::U8, DataType::S8, DataType::QASYMM8,
                                                         DataType::U16, DataType::S16,
                                                         DataType::U32, DataType::S32,
                                                         DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(output);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG((start == end), "start of the requested sequence must not be equal to the end");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(((start < end) && (step <= 0)), "step must be greater than 0 when start < end");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(((start > end) && (step >= 0)), "step must be less than 0 when start > end");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG((start == end), "start of the requested sequence must not be equal to the end");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!check_value_range(start, output->data_type(), output->quantization_info()), "start value is outside the range of the data type");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!check_value_range(end, output->data_type(), output->quantization_info()), "end value is outside the range of the data type");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!check_value_range(step, output->data_type(), output->quantization_info()), "step value is outside the range of the data type");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->num_dimensions() != 1, "Output has to be a 1-D tensor");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->tensor_shape().total_size() < num_of_elements_in_range(start, end, step), "Output tensor size is incorrect");

    return Status{};
}
} // namespace

CLRangeKernel::CLRangeKernel()
    : _start(0), _end(1), _step(1), _output(nullptr)
{
}

void CLRangeKernel::configure(ICLTensor *output, const float start, const float end, const float step)
{
    configure(CLKernelLibrary::get().get_compile_context(), output, start, end, step);
}

void CLRangeKernel::configure(const CLCompileContext &compile_context, ICLTensor *output, const float start, const float end, const float step)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(output->info(), start, end, step));

    // Configure kernel window
    unsigned int num_elems_processed_per_iteration = adjust_vec_size(vector_size_byte_opencl / output->info()->element_size(), output->info()->dimension(0));
    Window       win                               = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration));

    auto padding_info = get_padding_info({ output });

    _start  = start;
    _end    = end;
    _step   = step;
    _output = output;

    std::string kernel_name = "range";

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(output->info()->data_type()));
    build_opts.add_option("-DVECTOR_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration));
    build_opts.add_option("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(output->info()->dimension(0) % num_elems_processed_per_iteration));
    build_opts.add_option("-DSTART=" + support::cpp11::to_string(start));
    build_opts.add_option("-DSTEP=" + support::cpp11::to_string(step));
    if(is_data_type_quantized_asymmetric(output->info()->data_type()))
    {
        const UniformQuantizationInfo qinfo = output->info()->quantization_info().uniform();
        build_opts.add_option("-DOFFSET_OUT=" + support::cpp11::to_string(qinfo.offset));
        build_opts.add_option("-DSCALE_OUT=" + float_to_string_with_full_precision(qinfo.scale));
        kernel_name += "_quantized";
    }
    // Create kernel
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());
    ICLKernel::configure_internal(win);

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(output->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(0));
    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status CLRangeKernel::validate(const ITensorInfo *output, const float start, const float end, const float step)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(output, start, end, step));
    return Status{};
}

void CLRangeKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);
    unsigned int idx = 0;
    add_1D_tensor_argument(idx, _output, window);

    enqueue(queue, *this, window, lws_hint());
}
} // namespace arm_compute