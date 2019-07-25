/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLBatchConcatenateLayerKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Window.h"

#include "support/ToolchainSupport.h"

#include <map>

using namespace arm_compute;

namespace
{
std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, unsigned int batch_offset, ITensorInfo *output)
{
    ARM_COMPUTE_UNUSED(batch_offset);

    const unsigned int num_elems_processed_per_iteration = 16 / input->element_size();

    // The window needs to be based on output, except for the batch size
    Window win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration));
    // The total batch size is the concatenation of the batch size of the inputs
    win.set(3, Window::Dimension(0, input->tensor_shape()[3], 1));

    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
    bool                   window_changed = update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
Status validate_arguments(const ITensorInfo *input, unsigned int batch_offset, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S8, DataType::QASYMM8,
                                                         DataType::U16, DataType::S16,
                                                         DataType::U32, DataType::S32,
                                                         DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(Window::DimX) != output->dimension(Window::DimX));
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(Window::DimY) != output->dimension(Window::DimY));
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(Window::DimZ) != output->dimension(Window::DimZ));
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(3) + batch_offset > output->dimension(3));
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(4, input, output);

    return Status{};
}
} // namespace

CLBatchConcatenateLayerKernel::CLBatchConcatenateLayerKernel()
    : _input(nullptr), _output(nullptr), _batch_offset(0)
{
}

void CLBatchConcatenateLayerKernel::configure(const ICLTensor *input, unsigned int batch_offset, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), batch_offset, output->info()));

    _input        = input;
    _output       = output;
    _batch_offset = batch_offset;

    const unsigned int num_elems_processed_per_iteration = 16 / input->info()->element_size();

    // Add build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_underlying_cl_type_from_data_type(input->info()->data_type()));
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration));
    if(is_data_type_quantized_asymmetric(input->info()->data_type()) && input->info()->quantization_info() != output->info()->quantization_info())
    {
        const UniformQuantizationInfo iq_info = input->info()->quantization_info().uniform();
        const UniformQuantizationInfo oq_info = output->info()->quantization_info().uniform();

        build_opts.add_option("-DOFFSET_IN1=" + float_to_string_with_full_precision(iq_info.offset));
        build_opts.add_option("-DOFFSET_OUT=" + float_to_string_with_full_precision(oq_info.offset));
        build_opts.add_option("-DSCALE_IN1=" + float_to_string_with_full_precision(iq_info.scale));
        build_opts.add_option("-DSCALE_OUT=" + float_to_string_with_full_precision(oq_info.scale));
    }

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("concatenate", build_opts.options()));

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), batch_offset, output->info());
    ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

    ICLKernel::configure_internal(std::get<1>(win_config));

    // Set output valid region
    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));

    // Set config_id for enabling LWS tuning
    _config_id = "concatenate_";
    _config_id += support::cpp11::to_string(3);
    _config_id += "_";
    _config_id += support::cpp11::to_string(batch_offset);
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(2));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(3));
}

Status CLBatchConcatenateLayerKernel::validate(const arm_compute::ITensorInfo *input,
                                               unsigned int                    batch_offset,
                                               const arm_compute::ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, batch_offset, output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), batch_offset, output->clone().get()).first);
    return Status{};
}

void CLBatchConcatenateLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice = window.first_slice_window_3D();

    const int offset_to_first_elements_in_bytes = _batch_offset * _output->info()->strides_in_bytes()[3];

    unsigned int idx = 2 * num_arguments_per_3D_tensor(); // Skip the input and output parameters
    _kernel.setArg<cl_int>(idx, offset_to_first_elements_in_bytes);

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(window.slide_window_slice_3D(slice));
}
