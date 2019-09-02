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
#include "arm_compute/core/CL/kernels/CLFuseBatchNormalizationKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Window.h"

#include "support/ToolchainSupport.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input_weights, const ITensorInfo *bn_mean, const ITensorInfo *bn_var,
                          const ITensorInfo *fused_weights, const ITensorInfo *fused_bias,
                          const ITensorInfo *input_bias, const ITensorInfo *bn_beta, const ITensorInfo *bn_gamma,
                          float epsilon, FuseBatchNormalizationType fbn_type)
{
    ARM_COMPUTE_UNUSED(epsilon);
    ARM_COMPUTE_ERROR_ON_NULLPTR(input_weights, bn_mean, bn_var);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input_weights);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input_weights, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(bn_mean, bn_var);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input_weights, bn_mean, bn_var);
    ARM_COMPUTE_RETURN_ERROR_ON(input_bias == nullptr && fused_bias == nullptr);
    ARM_COMPUTE_RETURN_ERROR_ON(bn_mean->num_dimensions() > 1);

    if(fbn_type == FuseBatchNormalizationType::CONVOLUTION)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(input_weights->dimension(3) != bn_mean->dimension(0));
    }
    else
    {
        const size_t channel_idx = get_data_layout_dimension_index(input_weights->data_layout(), DataLayoutDimension::CHANNEL);
        ARM_COMPUTE_RETURN_ERROR_ON(input_weights->dimension(channel_idx) != bn_mean->dimension(0));
    }

    // Validate bias
    if(input_bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(bn_mean, input_bias);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input_weights, input_bias);
    }
    // Validate beta
    if(bn_beta != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(bn_mean, bn_beta);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input_weights, bn_beta);
    }
    // Validate gamma
    if(bn_gamma != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(bn_mean, bn_gamma);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input_weights, bn_gamma);
    }
    // Validate output weights
    if(fused_weights != nullptr && fused_weights->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input_weights, fused_weights);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input_weights, fused_weights);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input_weights, fused_weights);
    }
    // Validate output bias
    if(fused_bias != nullptr && fused_bias->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(bn_mean, fused_bias);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input_weights, fused_bias);
    }

    return Status{};
}
} // namespace

CLFuseBatchNormalizationKernel::CLFuseBatchNormalizationKernel()
    : _input_weights(nullptr), _input_bias(nullptr), _bn_mean(nullptr), _bn_var(nullptr), _bn_gamma(nullptr), _bn_beta(nullptr), _fused_weights(nullptr), _fused_bias(nullptr), _epsilon(),
      _run_in_place_weights(false), _run_in_place_bias(false)
{
}

void CLFuseBatchNormalizationKernel::configure(const ICLTensor *input_weights, const ICLTensor *bn_mean, const ICLTensor *bn_var,
                                               ICLTensor *fused_weights, ICLTensor *fused_bias,
                                               const ICLTensor *input_bias, const ICLTensor *bn_beta, const ICLTensor *bn_gamma,
                                               float epsilon, FuseBatchNormalizationType fbn_type)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input_weights, bn_mean, bn_var);

    _input_weights = input_weights;
    _input_bias    = input_bias;
    _bn_mean       = bn_mean;
    _bn_var        = bn_var;
    _bn_beta       = bn_beta;
    _bn_gamma      = bn_gamma;
    _fused_weights = fused_weights;
    _fused_bias    = fused_bias;
    _epsilon       = epsilon;

    _run_in_place_weights = (fused_weights == nullptr) || (fused_weights == input_weights);
    _run_in_place_bias    = (input_bias != nullptr && fused_bias == nullptr) || (input_bias != nullptr && fused_bias == input_bias);

    // Auto initialize outputs
    if(_fused_weights != nullptr)
    {
        // Output tensor auto initialization if not yet initialized
        auto_init_if_empty(*_fused_weights->info(), *_input_weights->info()->clone());
    }
    if(_fused_bias != nullptr)
    {
        // Output tensor auto initialization if not yet initialized
        auto_init_if_empty(*_fused_bias->info(), *_bn_mean->info()->clone());
    }

    // Validate arguments
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input_weights->info(), bn_mean->info(), bn_var->info(),
                                                  (fused_weights != nullptr) ? fused_weights->info() : nullptr,
                                                  (fused_bias != nullptr) ? fused_bias->info() : nullptr,
                                                  (input_bias != nullptr) ? input_bias->info() : nullptr,
                                                  (bn_beta != nullptr) ? bn_beta->info() : nullptr,
                                                  (bn_gamma != nullptr) ? bn_gamma->info() : nullptr,
                                                  epsilon, fbn_type));

    // Configure kernel window
    Window win = calculate_max_window(*input_weights->info());
    ICLKernel::configure_internal(win);

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input_weights->info()->data_type()));
    build_opts.add_option_if(fbn_type == FuseBatchNormalizationType::CONVOLUTION, "-DDIM2=" + support::cpp11::to_string(input_weights->info()->dimension(2)));
    build_opts.add_option("-DEPSILON=" + float_to_string_with_full_precision(epsilon));
    build_opts.add_option_if(_input_weights->info()->data_layout() == DataLayout::NHWC, "-DNHWC");
    build_opts.add_option_if(_run_in_place_weights, "-DIN_PLACE_W");
    build_opts.add_option_if(_run_in_place_bias, "-DIN_PLACE_B");
    build_opts.add_option_if(input_bias != nullptr, "-DBIAS");
    build_opts.add_option_if(bn_beta != nullptr, "-DBETA");
    build_opts.add_option_if(bn_gamma != nullptr, "-DGAMMA");

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("fuse_batchnormalization_layer", build_opts.options()));
}

Status CLFuseBatchNormalizationKernel::validate(const ITensorInfo *input_weights, const ITensorInfo *bn_mean, const ITensorInfo *bn_var,
                                                const ITensorInfo *fused_weights, const ITensorInfo *fused_bias,
                                                const ITensorInfo *input_bias, const ITensorInfo *bn_beta, const ITensorInfo *bn_gamma,
                                                float epsilon, FuseBatchNormalizationType fbn_type)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input_weights, bn_mean, bn_var, fused_weights, fused_bias, input_bias, bn_beta, bn_gamma, epsilon, fbn_type));
    return Status{};
}

void CLFuseBatchNormalizationKernel::run(const arm_compute::Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    // Create window slice
    Window collapsed_window = window.collapse(window, Window::DimZ);
    Window slice_1d         = window.first_slice_window_1D();
    Window slice_3d         = collapsed_window.first_slice_window_3D();

    // Add kernel arguments
    unsigned int idx = 0;
    add_3D_tensor_argument(idx, _input_weights, slice_3d);
    if(_input_bias != nullptr)
    {
        add_1D_tensor_argument(idx, _input_bias, slice_1d);
    }
    add_1D_tensor_argument(idx, _bn_mean, slice_1d);
    add_1D_tensor_argument(idx, _bn_var, slice_1d);
    if(!_run_in_place_weights)
    {
        add_3D_tensor_argument(idx, _fused_weights, slice_3d);
    }
    if(!_run_in_place_bias)
    {
        add_1D_tensor_argument(idx, _fused_bias, slice_1d);
    }
    if(_bn_beta != nullptr)
    {
        add_1D_tensor_argument(idx, _bn_beta, slice_1d);
    }
    if(_bn_gamma != nullptr)
    {
        add_1D_tensor_argument(idx, _bn_gamma, slice_1d);
    }
    enqueue(queue, *this, slice_3d);
}
} // namespace arm_compute
