/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLDirectConvolutionLayerOutputStageKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"

#include <cstddef>
#include <cstdint>

using namespace arm_compute;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::S32, DataType::F16,
                                                         DataType::F32);

    if(bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(bias);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(bias, 1, DataType::S32, DataType::F16, DataType::F32);

        if(is_data_type_quantized_asymmetric(input->data_type()))
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(bias, 1, DataType::S32);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, bias);
        }

        ARM_COMPUTE_RETURN_ERROR_ON(bias->num_dimensions() > 1);
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(is_data_type_float(input->data_type()),
                                        "Calling output stage kernel with floating point arguments");
    }

    // Checks performed on output
    if(input->data_type() == DataType::S32)
    {
        // Quantized configuration checks
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::QASYMM8);
    }
    else
    {
        // In case of out-of-place computation (supported for non-quantized configurations)
        if((output != nullptr) && (output->total_size() != 0))
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        }
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *bias, ITensorInfo *output)
{
    bool         window_changed                    = false;
    unsigned int num_elems_processed_per_iteration = 16 / element_size_from_data_type(input->data_type());

    // Configure kernel window
    Window win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));

    // Input window
    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
    window_changed = window_changed || update_window_and_padding(win, input_access);

    // Bias window
    if(bias != nullptr)
    {
        AccessWindowStatic bias_access(bias, 0, 0, ceil_to_multiple(bias->dimension(0), num_elems_processed_per_iteration), bias->dimension(1));
        window_changed = window_changed || update_window_and_padding(win, bias_access);
    }

    // Output window
    if(output != nullptr && (output->total_size() != 0))
    {
        AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
        window_changed = window_changed || update_window_and_padding(win, output_access);
        output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));
    }
    else
    {
        input_access.set_valid_region(win, ValidRegion(Coordinates(), input->tensor_shape()));
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

CLDirectConvolutionLayerOutputStageKernel::CLDirectConvolutionLayerOutputStageKernel()
    : _input(nullptr), _bias(nullptr), _output(nullptr), _result_fixedpoint_multiplier(0), _result_shift(0), _result_offset_after_shift(0)
{
}

void CLDirectConvolutionLayerOutputStageKernel::configure(ICLTensor *input, const ICLTensor *bias, ICLTensor *output,
                                                          int result_fixedpoint_multiplier, int result_shift, int result_offset_after_shift)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input);

    // Auto-initialize output if required
    if(output != nullptr)
    {
        // Work out expected output data type
        const DataType output_dt = (input->info()->data_type() == DataType::S32) ? DataType::QASYMM8 : input->info()->data_type();
        // Output tensor auto initialization if not yet initialized
        auto_init_if_empty(*output->info(), input->info()->clone()->set_data_type(output_dt));
    }

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), (bias == nullptr) ? nullptr : bias->info(), (output == nullptr) ? nullptr : output->info()));

    _bias                         = bias;
    _input                        = input;
    _output                       = output;
    _result_fixedpoint_multiplier = result_fixedpoint_multiplier;
    _result_shift                 = result_shift;
    _result_offset_after_shift    = result_offset_after_shift;

    const unsigned int num_elems_accessed_per_iteration = 16 / element_size_from_data_type(input->info()->data_type());

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option_if(bias != nullptr, "-DHAS_BIAS");
    build_opts.add_option("-D" + string_from_data_layout(input->info()->data_layout()));
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_accessed_per_iteration));
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("output_stage_quantized", build_opts.options()));

    // Set static kernel arguments
    int idx = 2 * num_arguments_per_3D_tensor() + ((bias != nullptr) ? num_arguments_per_1D_tensor() : 0);
    _kernel.setArg<int>(idx++, _result_offset_after_shift);
    _kernel.setArg<int>(idx++, _result_fixedpoint_multiplier);
    _kernel.setArg<int>(idx++, _result_shift);

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), (bias == nullptr) ? nullptr : bias->info(), (output == nullptr) ? nullptr : output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);
}

Status CLDirectConvolutionLayerOutputStageKernel::validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, bias, output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), bias == nullptr ? nullptr : bias->clone().get(), output == nullptr ? nullptr : output->clone().get()).first);

    return Status{};
}

void CLDirectConvolutionLayerOutputStageKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

    Window slice = window.first_slice_window_3D();

    // Set bias vector
    if(_bias != nullptr)
    {
        unsigned int idx1 = 2 * num_arguments_per_3D_tensor();
        Window       slice_biases;
        slice_biases.use_tensor_dimensions(_bias->info()->tensor_shape());
        add_1D_tensor_argument(idx1, _bias, slice_biases);
    }

    // Run kernel
    do
    {
        // Set arguments
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(window.slide_window_slice_3D(slice));
}
