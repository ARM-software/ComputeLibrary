/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/core/GLES_COMPUTE/kernels/GCBatchNormalizationLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/GLES_COMPUTE/GCHelpers.h"
#include "arm_compute/core/GLES_COMPUTE/GCKernelLibrary.h"
#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
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
    ARM_COMPUTE_UNUSED(var);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, mean, var);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, mean, var);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(mean, var);

    if(output->total_size() != 0)
    {
        ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);
    }

    if(beta != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(mean, beta);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, beta);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT(input, beta);
    }
    if(gamma != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(mean, gamma);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, gamma);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT(input, gamma);
    }
    if(act_info.enabled())
    {
        ARM_COMPUTE_ERROR_ON(input->data_type() != DataType::F32 && input->data_type() != DataType::F16);
        ARM_COMPUTE_ERROR_ON(act_info.activation() != ActivationLayerInfo::ActivationLayerInfo::ActivationFunction::RELU
                             && act_info.activation() != ActivationLayerInfo::ActivationLayerInfo::ActivationFunction::BOUNDED_RELU
                             && act_info.activation() != ActivationLayerInfo::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU);
        ARM_COMPUTE_ERROR_ON(act_info.b() > act_info.a());
    }
    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output,
                                                        ITensorInfo *mean, ITensorInfo *var,
                                                        ITensorInfo *beta, ITensorInfo *gamma)
{
    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*output, input->tensor_shape(), 1, input->data_type(), input->fixed_point_position());

    unsigned int num_elems_processed_per_iteration = 1;
    if(input->data_type() == DataType::F16)
    {
        num_elems_processed_per_iteration = 4;
    }

    // Configure kernel window
    Window win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
    AccessWindowStatic     mean_access(mean, 0, 0, mean->dimension(0) + 3, mean->dimension(1));
    AccessWindowStatic     var_access(var, 0, 0, var->dimension(0) + 3, var->dimension(1));

    bool window_changed = false;
    if(beta != nullptr)
    {
        AccessWindowStatic beta_access(beta, 0, 0, beta->dimension(0) + 3, beta->dimension(1));
        if(gamma != nullptr)
        {
            AccessWindowStatic gamma_access(gamma, 0, 0, gamma->dimension(0) + 3, gamma->dimension(1));
            window_changed = update_window_and_padding(win, input_access, output_access, mean_access, var_access, beta_access, gamma_access);
        }
        else
        {
            window_changed = update_window_and_padding(win, input_access, output_access, mean_access, var_access, beta_access);
        }
    }
    else
    {
        if(gamma != nullptr)
        {
            AccessWindowStatic gamma_access(gamma, 0, 0, gamma->dimension(0) + 3, gamma->dimension(1));
            window_changed = update_window_and_padding(win, input_access, output_access, mean_access, var_access, gamma_access);
        }
        else
        {
            window_changed = update_window_and_padding(win, input_access, output_access, mean_access, var_access);
        }
    }
    output_access.set_valid_region(win, input->valid_region());

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

GCBatchNormalizationLayerKernel::GCBatchNormalizationLayerKernel()
    : _input(nullptr), _output(nullptr), _mean(nullptr), _var(nullptr), _beta(nullptr), _gamma(nullptr), _epsilon(0.0f)
{
}

void GCBatchNormalizationLayerKernel::configure(const IGCTensor *input, IGCTensor *output, const IGCTensor *mean, const IGCTensor *var, const IGCTensor *beta, const IGCTensor *gamma,
                                                float epsilon, ActivationLayerInfo act_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, mean, var);

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), mean->info(), var->info(),
                                                  (beta != nullptr) ? beta->info() : nullptr, (gamma != nullptr) ? gamma->info() : nullptr,
                                                  epsilon, act_info));

    _input   = input;
    _output  = output;
    _mean    = mean;
    _var     = var;
    _beta    = beta;
    _gamma   = gamma;
    _epsilon = epsilon;

    // Set build options
    std::set<std::string> build_opts;
    std::string           dt_name = (input->info()->data_type() == DataType::F32) ? "DATA_TYPE_FP32" : "DATA_TYPE_FP16";
    build_opts.emplace(("#define " + dt_name));
    build_opts.emplace(("#define ESPILON " + float_to_string_with_full_precision(_epsilon)));
    build_opts.emplace(("#define LOCAL_SIZE_X " + support::cpp11::to_string(1)));
    build_opts.emplace(("#define LOCAL_SIZE_Y " + support::cpp11::to_string(1)));
    build_opts.emplace(("#define LOCAL_SIZE_Z " + support::cpp11::to_string(1)));
    if(beta == nullptr)
    {
        build_opts.emplace("#define USE_DEFAULT_BETA");
    }
    if(gamma == nullptr)
    {
        build_opts.emplace("#define USE_DEFAULT_GAMMA");
    }

    if(act_info.enabled())
    {
        build_opts.emplace("#define " + string_from_activation_func(act_info.activation()));
        build_opts.emplace("#define A_VAL " + float_to_string_with_full_precision(act_info.a()));
        build_opts.emplace("#define B_VAL " + float_to_string_with_full_precision(act_info.b()));
    }

    // Create kernel
    _kernel = static_cast<GCKernel>(GCKernelLibrary::get().create_kernel("batchnormalization_layer", build_opts));

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), mean->info(), var->info(),
                                                    (beta != nullptr) ? beta->info() : nullptr, (gamma != nullptr) ? gamma->info() : nullptr);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);

    IGCKernel::configure(win_config.second);
}

Status GCBatchNormalizationLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output,
                                                 const ITensorInfo *mean, const ITensorInfo *var,
                                                 const ITensorInfo *beta, const ITensorInfo *gamma,
                                                 float epsilon, ActivationLayerInfo act_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, mean, var, beta, gamma, epsilon, act_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(),
                                                              mean->clone().get(), var->clone().get(),
                                                              beta->clone().get(), gamma->clone().get())
                                .first);

    return Status{};
}

void GCBatchNormalizationLayerKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    _kernel.use();

    _output->set_needs_shifting(true);

    Window slice    = window.first_slice_window_3D();
    Window slice_in = window.first_slice_window_3D();

    Window vector_slice = window.first_slice_window_1D();
    vector_slice.set(Window::DimX, Window::Dimension(0, 0, 0));

    unsigned int idx           = 2 * num_arguments_per_3D_tensor();
    unsigned int binding_point = 3;
    add_1D_tensor_argument(idx, _mean, binding_point, vector_slice);
    add_1D_tensor_argument(idx, _var, ++binding_point, vector_slice);
    if(_beta != nullptr)
    {
        add_1D_tensor_argument(idx, _beta, ++binding_point, vector_slice);
    }
    if(_gamma != nullptr)
    {
        add_1D_tensor_argument(idx, _gamma, ++binding_point, vector_slice);
    }

    slice.shift(Window::DimX, -(_output->info()->padding()).left);

    do
    {
        idx = 0;
        add_3D_tensor_argument(idx, _input, 1, slice_in);
        add_3D_tensor_argument(idx, _output, 2, slice);

        _kernel.update_shader_params();
        enqueue(*this, slice);
    }
    while(window.slide_window_slice_3D(slice) && window.slide_window_slice_3D(slice_in));
}
