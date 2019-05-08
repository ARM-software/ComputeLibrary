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
Status validate_arguments(const ITensorInfo *conv_weights, const ITensorInfo *bn_mean, const ITensorInfo *bn_var,
                          const ITensorInfo *fused_weights, const ITensorInfo *fused_bias,
                          const ITensorInfo *conv_bias, const ITensorInfo *bn_beta, const ITensorInfo *bn_gamma,
                          float epsilon)
{
    ARM_COMPUTE_UNUSED(epsilon);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(conv_weights);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(conv_weights, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(bn_mean, bn_var);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(conv_weights, bn_mean, bn_var);

    unsigned int kernels_idx = get_data_layout_dimension_index(conv_weights->data_layout(), DataLayoutDimension::BATCHES);
    ARM_COMPUTE_RETURN_ERROR_ON(conv_weights->dimension(kernels_idx) != bn_mean->dimension(0));

    // Validate bias
    if(conv_bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(bn_mean, conv_bias);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(conv_weights, conv_bias);
    }
    // Validate beta
    if(bn_beta != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(bn_mean, bn_beta);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(conv_weights, bn_beta);
    }
    // Validate gamma
    if(bn_gamma != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(bn_mean, bn_gamma);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(conv_weights, bn_gamma);
    }

    // Validate output weights
    if(fused_weights != nullptr && fused_weights->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(conv_weights, fused_weights);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(conv_weights, fused_weights);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(conv_weights, fused_weights);
    }
    // Validate output bias
    if(fused_bias != nullptr && fused_bias->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(bn_mean, fused_bias);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(conv_weights, fused_bias);
    }

    return Status{};
}
} // namespace

CLFuseBatchNormalizationKernel::CLFuseBatchNormalizationKernel()
    : _conv_weights(nullptr), _conv_bias(nullptr), _bn_mean(nullptr), _bn_var(nullptr), _bn_gamma(nullptr), _bn_beta(nullptr), _fused_weights(nullptr), _fused_bias(nullptr), _epsilon(),
      _run_in_place_weights(false), _run_in_place_bias(false)
{
}

void CLFuseBatchNormalizationKernel::configure(const ICLTensor *conv_weights, const ICLTensor *bn_mean, const ICLTensor *bn_var,
                                               ICLTensor *fused_weights, ICLTensor *fused_bias,
                                               const ICLTensor *conv_bias, const ICLTensor *bn_beta, const ICLTensor *bn_gamma,
                                               float epsilon)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(conv_weights, bn_mean, bn_var);

    _conv_weights  = conv_weights;
    _conv_bias     = conv_bias;
    _bn_mean       = bn_mean;
    _bn_var        = bn_var;
    _bn_beta       = bn_beta;
    _bn_gamma      = bn_gamma;
    _fused_weights = fused_weights;
    _fused_bias    = fused_bias;
    _epsilon       = epsilon;

    _run_in_place_weights = (fused_weights == nullptr) || (fused_weights == conv_weights);
    _run_in_place_bias    = (fused_bias == nullptr) || (conv_bias != nullptr && fused_bias == conv_bias);

    // Auto initialize outputs
    if(_fused_weights != nullptr)
    {
        // Output tensor auto initialization if not yet initialized
        auto_init_if_empty(*_fused_weights->info(), *_conv_weights->info()->clone());
        fused_weights->info()->set_valid_region(conv_weights->info()->valid_region());
    }
    if(_fused_bias != nullptr)
    {
        // Output tensor auto initialization if not yet initialized
        auto_init_if_empty(*_fused_bias->info(), *_bn_mean->info()->clone());
        _fused_bias->info()->set_valid_region(bn_mean->info()->valid_region());
    }

    // Validate arguments
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(conv_weights->info(), bn_mean->info(), bn_var->info(),
                                                  (fused_weights != nullptr) ? fused_weights->info() : nullptr,
                                                  (fused_bias != nullptr) ? fused_bias->info() : nullptr,
                                                  (conv_bias != nullptr) ? conv_bias->info() : nullptr,
                                                  (bn_beta != nullptr) ? bn_beta->info() : nullptr,
                                                  (bn_gamma != nullptr) ? bn_gamma->info() : nullptr,
                                                  epsilon));

    // Configure kernel window
    const unsigned int num_elems_processed_per_iteration_x = 4;
    const int          output_width_x                      = conv_weights->info()->tensor_shape().x();
    const bool         multi_access_x                      = (output_width_x / num_elems_processed_per_iteration_x > 0);

    Window win = calculate_max_window(*conv_weights->info());
    if(multi_access_x)
    {
        win.set(Window::DimX, Window::Dimension(win.x().start(),
                                                ceil_to_multiple(win.x().end(), num_elems_processed_per_iteration_x),
                                                num_elems_processed_per_iteration_x));
    }
    ICLKernel::configure_internal(win);

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(conv_weights->info()->data_type()));
    build_opts.add_option("-DSELECT_DATA_TYPE=" + get_cl_select_type_from_data_type(conv_weights->info()->data_type()));
    build_opts.add_option("-DNUM_CHANNELS=" + support::cpp11::to_string(conv_weights->info()->dimension(2)));
    build_opts.add_option("-DEPSILON=" + float_to_string_with_full_precision(epsilon));
    build_opts.add_option_if(multi_access_x, "-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration_x));
    build_opts.add_option_if(multi_access_x, "-DLAST_ACCESSED_X=" + support::cpp11::to_string(std::max<int>(output_width_x - num_elems_processed_per_iteration_x, 0)));
    build_opts.add_option_if(_run_in_place_weights, "-DIN_PLACE_W");
    build_opts.add_option_if(_run_in_place_bias, "-DIN_PLACE_B");
    build_opts.add_option_if(conv_bias != nullptr, "-DHAS_BIAS");
    build_opts.add_option_if(bn_beta == nullptr, "-DUSE_DEFAULT_BETA");
    build_opts.add_option_if(bn_gamma == nullptr, "-DUSE_DEFAULT_GAMMA");

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("fuse_batchnormalization_layer", build_opts.options()));
}

Status CLFuseBatchNormalizationKernel::validate(const ITensorInfo *conv_weights, const ITensorInfo *bn_mean, const ITensorInfo *bn_var,
                                                const ITensorInfo *fused_weights, const ITensorInfo *fused_bias,
                                                const ITensorInfo *conv_bias, const ITensorInfo *bn_beta, const ITensorInfo *bn_gamma,
                                                float epsilon)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(conv_weights, bn_mean, bn_var, fused_weights, fused_bias, conv_bias, bn_beta, bn_gamma, epsilon));
    return Status{};
}

void CLFuseBatchNormalizationKernel::run(const arm_compute::Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    // Create window slice
    Window collapsed_window = window.collapse_if_possible(window, Window::DimZ);
    Window slice            = collapsed_window.first_slice_window_4D();

    Window vector_slice = window.first_slice_window_1D();
    vector_slice.set(Window::DimX, Window::Dimension(0, 0, 0));

    // Add kernel arguments
    unsigned int idx = 0;
    add_4D_tensor_argument(idx, _conv_weights, slice);
    add_1D_tensor_argument(idx, _bn_mean, vector_slice);
    add_1D_tensor_argument(idx, _bn_var, vector_slice);
    if(!_run_in_place_weights)
    {
        add_4D_tensor_argument(idx, _fused_weights, slice);
    }
    if(!_run_in_place_bias)
    {
        add_1D_tensor_argument(idx, _fused_bias, vector_slice);
    }
    if(_conv_bias != nullptr)
    {
        add_1D_tensor_argument(idx, _conv_bias, vector_slice);
    }
    if(_bn_beta != nullptr)
    {
        add_1D_tensor_argument(idx, _bn_beta, vector_slice);
    }
    if(_bn_gamma != nullptr)
    {
        add_1D_tensor_argument(idx, _bn_gamma, vector_slice);
    }
    enqueue(queue, *this, slice);
}
} // namespace arm_compute
