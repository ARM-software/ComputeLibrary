/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLBatchNormalizationLayerKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Window.h"

#include "support/ToolchainSupport.h"

using namespace arm_compute;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output,
                          const ITensorInfo *mean, const ITensorInfo *var,
                          const ITensorInfo *beta, const ITensorInfo *gamma,
                          float epsilon, ActivationLayerInfo act_info)
{
    ARM_COMPUTE_UNUSED(epsilon);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(mean, var);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, mean, var);
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::CHANNEL)) != mean->dimension(0));
    if(beta != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(mean, beta);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, beta);
    }
    if(gamma != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(mean, gamma);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, gamma);
    }

    if(act_info.enabled())
    {
        ActivationLayerInfo::ActivationFunction act = act_info.activation();
        ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() != DataType::F32 && input->data_type() != DataType::F16);
        ARM_COMPUTE_RETURN_ERROR_ON(act != ActivationLayerInfo::ActivationLayerInfo::ActivationFunction::RELU
                                    && act != ActivationLayerInfo::ActivationLayerInfo::ActivationFunction::BOUNDED_RELU
                                    && act != ActivationLayerInfo::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU);
        ARM_COMPUTE_RETURN_ERROR_ON(act_info.b() > act_info.a());
    }

    if(output != nullptr && output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output,
                                                        ITensorInfo *mean, ITensorInfo *var, ITensorInfo *beta, ITensorInfo *gamma)
{
    if(output != nullptr)
    {
        // Output tensor auto initialization if not yet initialized
        auto_init_if_empty(*output, *input->clone());
    }

    const unsigned int num_elems_processed_per_iteration = 16 / input->element_size();

    // Configure kernel window
    Window                 win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);

    bool window_changed = false;
    if(output != nullptr)
    {
        AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
        window_changed = update_window_and_padding(win, input_access, output_access);
        output_access.set_valid_region(win, input->valid_region());
    }
    else
    {
        window_changed = update_window_and_padding(win, input_access);
    }

    // Mean, var, gamma and beta get parallelized for the NHWC case as they follow the channel dimension, which is along the first axis
    if(input->data_layout() == DataLayout::NHWC)
    {
        AccessWindowHorizontal mean_access(mean, 0, num_elems_processed_per_iteration);
        AccessWindowHorizontal var_access(var, 0, num_elems_processed_per_iteration);
        window_changed = window_changed || update_window_and_padding(win, mean_access, var_access);

        if(beta != nullptr)
        {
            AccessWindowHorizontal beta_access(beta, 0, num_elems_processed_per_iteration);
            window_changed = window_changed || update_window_and_padding(win, beta_access);
        }
        if(gamma != nullptr)
        {
            AccessWindowHorizontal gamma_access(gamma, 0, num_elems_processed_per_iteration);
            window_changed = window_changed || update_window_and_padding(win, gamma_access);
        }
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

CLBatchNormalizationLayerKernel::CLBatchNormalizationLayerKernel()
    : _input(nullptr), _output(nullptr), _mean(nullptr), _var(nullptr), _beta(nullptr), _gamma(nullptr), _epsilon(0), _run_in_place(false)
{
}

void CLBatchNormalizationLayerKernel::configure(ICLTensor *input, ICLTensor *output, const ICLTensor *mean, const ICLTensor *var, const ICLTensor *beta, const ICLTensor *gamma,
                                                float epsilon, ActivationLayerInfo act_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, mean, var);

    _input   = input;
    _output  = output;
    _mean    = mean;
    _var     = var;
    _beta    = beta;
    _gamma   = gamma;
    _epsilon = epsilon;

    _run_in_place = (output == nullptr) || (output == input);

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), (output != nullptr) ? output->info() : nullptr,
                                                  mean->info(), var->info(), (beta != nullptr) ? beta->info() : nullptr,
                                                  (gamma != nullptr) ? gamma->info() : nullptr, epsilon, act_info));

    const unsigned int num_elems_processed_per_iteration = 16 / input->info()->element_size();

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration));
    build_opts.add_option("-DACTIVATION_TYPE=" + lower_string(string_from_activation_func(act_info.activation())));
    build_opts.add_option_if(act_info.enabled(), "-DA_VAL=" + float_to_string_with_full_precision(act_info.a()));
    build_opts.add_option_if(act_info.enabled(), "-DB_VAL=" + float_to_string_with_full_precision(act_info.b()));
    build_opts.add_option_if(_run_in_place, "-DIN_PLACE");
    build_opts.add_option_if(beta == nullptr, "-DUSE_DEFAULT_BETA");
    build_opts.add_option_if(gamma == nullptr, "-DUSE_DEFAULT_GAMMA");

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("batchnormalization_layer_" + lower_string(string_from_data_layout(input->info()->data_layout())), build_opts.options()));

    // Set kernel static arguments
    unsigned int include_output = (!_run_in_place) ? 1 : 0;
    unsigned int idx            = (1 + include_output) * num_arguments_per_3D_tensor() + 2 * num_arguments_per_1D_tensor(); // Skip the input and output parameters
    if(_beta != nullptr)
    {
        idx += num_arguments_per_1D_tensor(); // Skip beta parameter
    }
    if(_gamma != nullptr)
    {
        idx += num_arguments_per_1D_tensor(); // Skip gamma parameter
    }
    _kernel.setArg<cl_float>(idx++, _epsilon);

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), (_run_in_place) ? nullptr : output->info(),
                                                    mean->info(), var->info(),
                                                    (beta != nullptr) ? beta->info() : nullptr,
                                                    (gamma != nullptr) ? gamma->info() : nullptr);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

    _config_id = "batch_normalization_layer_";
    _config_id += string_from_data_type(input->info()->data_type());
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(2));
    _config_id += "_";
    _config_id += lower_string(string_from_data_layout(input->info()->data_layout()));
}

Status CLBatchNormalizationLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output,
                                                 const ITensorInfo *mean, const ITensorInfo *var,
                                                 const ITensorInfo *beta, const ITensorInfo *gamma,
                                                 float epsilon, ActivationLayerInfo act_info)
{
    const bool run_in_place = (output == nullptr) || (output == input);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, mean, var, beta, gamma, epsilon, act_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), (run_in_place) ? nullptr : output->clone().get(),
                                                              mean->clone().get(), var->clone().get(),
                                                              (beta != nullptr) ? beta->clone().get() : nullptr,
                                                              (gamma != nullptr) ? gamma->clone().get() : nullptr)
                                .first);

    return Status{};
}

void CLBatchNormalizationLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    Window slice = window.first_slice_window_3D();

    Window vector_slice = window.first_slice_window_1D();
    vector_slice.set(Window::DimX, Window::Dimension(0, 0, 0));

    unsigned int include_output = (!_run_in_place) ? 1 : 0;
    unsigned int idx            = (1 + include_output) * num_arguments_per_3D_tensor();
    add_1D_tensor_argument(idx, _mean, vector_slice);
    add_1D_tensor_argument(idx, _var, vector_slice);
    if(_beta != nullptr)
    {
        add_1D_tensor_argument(idx, _beta, vector_slice);
    }
    if(_gamma != nullptr)
    {
        add_1D_tensor_argument(idx, _gamma, vector_slice);
    }

    do
    {
        idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        if(!_run_in_place)
        {
            add_3D_tensor_argument(idx, _output, slice);
        }
        enqueue(queue, *this, slice, lws_hint());
    }
    while(window.slide_window_slice_3D(slice));
}
