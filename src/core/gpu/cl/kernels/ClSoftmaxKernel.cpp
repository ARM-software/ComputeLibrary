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
#include "src/core/gpu/cl/kernels/ClSoftmaxKernel.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/experimental/Types.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/Cast.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
namespace
{
/** Calculates softmax parameters from the quantized input scale and scaling factor for the exponent and places them as build options.
 *
 * Prepares these build options:
 * -INPUT_BETA_MULTIPLIER, INPUT_BETA_LEFT_SHIFT - quantized representation of beta multiplier.
 * -DIFF_MIN - threshold difference between maximum value of input data and current processed value,
 *             it defines whether the value will be taken into account or not.
 *
 * @param[in] build_opts  Build options to extend
 * @param[in] input_scale Input scaling factor
 * @param[in] beta        Exponent scaling factor beta
 */
CLBuildOptions prepare_quantized_softmax_build_options(float input_scale, float beta)
{
    // Number of integer bits in temporary fixed-point representation of current-to-max difference
    static const int scaled_diff_int_bits = 5;
    // Number of integer bits used in temporary fixed-point representation of exponent accumulator
    static const int exp_accumulation_in_bits = 12;

    const double beta_multiplier = std::min(
                                       1.0 * beta * input_scale * (1 << (31 - scaled_diff_int_bits)),
                                       (1LL << 31) - 1.0);
    int input_beta_multiplier;
    int input_beta_left_shift;
    quantization::calculate_quantized_multiplier_greater_than_one(beta_multiplier, &input_beta_multiplier, &input_beta_left_shift);

    const double max_input_rescaled = 1.0 * ((1 << scaled_diff_int_bits) - 1) * (1LL << (31 - scaled_diff_int_bits)) / (1LL << input_beta_left_shift);
    const int    diff_min           = -1.f * std::floor(max_input_rescaled);

    CLBuildOptions build_opts;
    build_opts.add_option("-DSCALED_DIFF_INT_BITS=" + support::cpp11::to_string(scaled_diff_int_bits));
    build_opts.add_option("-DEXP_ACCUMULATION_INT_BITS=" + support::cpp11::to_string(exp_accumulation_in_bits));
    build_opts.add_option("-DINPUT_BETA_MULTIPLIER=" + support::cpp11::to_string(input_beta_multiplier));
    build_opts.add_option("-DINPUT_BETA_LEFT_SHIFT=" + support::cpp11::to_string(input_beta_left_shift));
    build_opts.add_option("-DDIFF_MIN=" + support::cpp11::to_string(diff_min));

    return build_opts;
}

Status validate_arguments_1DMaxShiftExpSum(const ITensorInfo &src, const ITensorInfo &max, const ITensorInfo &dst, const ITensorInfo &sum)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(&src);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src, &max);

    const bool is_quantized_asymmetric = is_data_type_quantized_asymmetric(src.data_type());

    // Checks performed when output is configured
    if(dst.total_size() != 0)
    {
        if(is_quantized_asymmetric)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&dst, 1, DataType::S32);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src, &dst);
        }
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&src, &dst);
    }

    // Checks performed when sum is configured
    if(sum.total_size() != 0)
    {
        if(is_quantized_asymmetric)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&sum, 1, DataType::S32);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&max, &sum);
        }
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&max, &sum);
    }

    return Status{};
}

Status validate_arguments_1DNorm(const ITensorInfo &src, const ITensorInfo &sum, const ITensorInfo &dst, const SoftmaxKernelInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(&src);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src, 1, DataType::S32, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src, &sum);
    ARM_COMPUTE_RETURN_ERROR_ON(info.is_log && !is_data_type_float(info.input_data_type));

    // Note: output should always have a scale of 1/256 and offset 0
    const QuantizationInfo allowed_quantization_info = get_softmax_output_quantization_info(info.input_data_type, info.is_log);
    const bool             is_quantized_asymmetric   = is_data_type_quantized_asymmetric(info.input_data_type);

    // Checks performed when output is configured
    if(dst.total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&src, &dst);
        if(!is_quantized_asymmetric)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src, &dst);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&dst, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED);
            ARM_COMPUTE_RETURN_ERROR_ON(dst.quantization_info() != allowed_quantization_info);
        }
    }

    return Status{};
}
} // namespace

/**< Grid size (obtained through auto-tuning) */
const unsigned int ClLogits1DMaxShiftExpSumKernel::_grid_size = 64;
/**< Vector size in the serial case (obtained through auto-tuning) */
const unsigned int ClLogits1DMaxShiftExpSumKernel::_serial_vector_size = 8;
/**< Vector size in the parallel case (obtained through auto-tuning, enables the best memory access pattern for Bifrost) .*/
const unsigned int ClLogits1DMaxShiftExpSumKernel::_parallel_vector_size = 4;

ClLogits1DMaxShiftExpSumKernel::ClLogits1DMaxShiftExpSumKernel()
{
    _type = CLKernelType::ELEMENTWISE;
}

void ClLogits1DMaxShiftExpSumKernel::configure(const CLCompileContext &compile_context, const ITensorInfo &src, ITensorInfo &max, ITensorInfo &dst, ITensorInfo &sum, const SoftmaxKernelInfo &info)
{
    auto padding_info = get_padding_info({ &src, &max, &dst, &sum });

    // Output auto initialization if not yet initialized
    auto_init_if_empty(sum, src.clone()->set_tensor_shape(max.tensor_shape()));
    auto_init_if_empty(dst, *src.clone());

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_1DMaxShiftExpSum(src, max, dst, sum));

    const DataType                dt                 = src.data_type();
    const UniformQuantizationInfo qinfo              = src.quantization_info().uniform();
    const size_t                  reduction_dim_size = src.dimension(0);
    const float                   beta               = info.beta;
    const auto                    is_signed_qasymm8  = is_data_type_quantized_asymmetric_signed(info.input_data_type);
    const int                     min_value          = is_signed_qasymm8 ? CL_SCHAR_MIN : 0;

    ParallelReductionInfo parallel_reduction_info = is_parallel_reduction(reduction_dim_size);
    const unsigned int    vector_size             = adjust_vec_size(std::get<1>(parallel_reduction_info), reduction_dim_size);

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(dt));
    build_opts.add_option("-DMIN_VALUE=" + support::cpp11::to_string(min_value));
    build_opts.add_option("-DVECTOR_SIZE=" + support::cpp11::to_string(vector_size));
    build_opts.add_option("-DSRC_WIDTH=" + support::cpp11::to_string(reduction_dim_size));
    build_opts.add_option("-DVECTOR_SIZE_LEFTOVER=" + support::cpp11::to_string(reduction_dim_size % vector_size));
    build_opts.add_option("-DLOG_VECTOR_SIZE=" + support::cpp11::to_string(lround(log2(vector_size))));
    build_opts.add_option_if((reduction_dim_size % vector_size) != 0, "-DNON_MULTIPLE_OF_VECTOR_SIZE");
    build_opts.add_option_if(is_signed_qasymm8, "-DQASYMM8_SIGNED");
    build_opts.add_option_if(is_data_type_float(dt) && (beta != 1.0f), "-DBETA=" + float_to_string_with_full_precision(beta));
    build_opts.add_option_if(is_data_type_float(dt) && info.is_log, "-DLOG_SOFTMAX");
    build_opts.add_option_if(is_data_type_float(dt), "-DMINVAL=" + ((dt == DataType::F16) ? std::string("-HALF_MAX") : std::string("-FLT_MAX")));
    build_opts.add_options_if(is_data_type_quantized_asymmetric(dt), prepare_quantized_softmax_build_options(qinfo.scale, beta).options());

    cl::NDRange lws_hint(cl::NullRange);
    std::string kernel_name = std::string("softmax_layer_max_shift_exp_sum_") + (is_data_type_quantized_asymmetric(dt) ? "quantized_" : "");

    // Configure parallel kernel if needed
    if(std::get<0>(parallel_reduction_info))
    {
        kernel_name += "parallel";
        bool is_grid_size_pow2 = (_grid_size != 0) && ((_grid_size & (_grid_size - 1)) == 0);
        build_opts.add_option_if(is_grid_size_pow2 && _grid_size <= 256, "-DGRID_SIZE=" + support::cpp11::to_string(_grid_size));

        // Handle boundary conditions.
        const unsigned int multiple_grid_size = (reduction_dim_size / vector_size) % _grid_size;
        build_opts.add_option_if((multiple_grid_size != 0) || ((reduction_dim_size % vector_size) != 0), "-DNON_MULTIPLE_OF_GRID_SIZE");
        // Setting _lws_hint in this way can also communicate grid_size to ClLogits1DMaxShiftExpSumKernel::run().
        // A single workgroup performs reduction in dimension 0 in the parallel case, hence lws[0]==gws[0].
        lws_hint = cl::NDRange(_grid_size);
    }
    else
    {
        kernel_name += "serial";
    }

    // Create kernel.
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Configure window
    Window win = calculate_max_window(src, Steps(reduction_dim_size));
    IClKernel::configure_internal(win, lws_hint);

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status ClLogits1DMaxShiftExpSumKernel::validate(const ITensorInfo &src, const ITensorInfo &max, const ITensorInfo &dst, const ITensorInfo &sum)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_1DMaxShiftExpSum(src, max, dst, sum));
    return Status{};
}

ClLogits1DMaxShiftExpSumKernel::ParallelReductionInfo ClLogits1DMaxShiftExpSumKernel::is_parallel_reduction(size_t size)
{
    bool         is_parallel_reduction = (size >= (_grid_size * _serial_vector_size)) && (_grid_size > 1);
    unsigned int vector_size           = is_parallel_reduction ? _parallel_vector_size : _serial_vector_size;
    return std::make_tuple(is_parallel_reduction, vector_size);
}

void ClLogits1DMaxShiftExpSumKernel::run_op(ITensorPack &tensors, const Window &window, ::cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    auto src = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    auto dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));
    auto max = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_INT_0));
    auto sum = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_INT_1));

    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst, max, sum);

    // Collapse window in Z dimension
    Window window_collapsed = window.collapse_if_possible(IClKernel::window(), Window::DimZ);

    // Reconfigure window in case of parallel reduction
    ParallelReductionInfo parallel_reduction_info = is_parallel_reduction(src->info()->dimension(0));
    if(std::get<0>(parallel_reduction_info))
    {
        // Launch grid_size parallel work items
        window_collapsed.set(Window::DimX, Window::Dimension(0, _grid_size, 1));
    }

    // Get slices
    Window slice = window_collapsed.first_slice_window_3D();
    do
    {
        unsigned int idx = 0;
        // Set inputs
        add_3D_tensor_argument(idx, src, slice);
        add_3D_tensor_argument(idx, max, slice);
        add_3D_tensor_argument(idx, dst, slice);
        add_3D_tensor_argument(idx, sum, slice);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(window_collapsed.slide_window_slice_3D(slice));
}

ClLogits1DNormKernel::ClLogits1DNormKernel()
{
    _type = CLKernelType::ELEMENTWISE;
}

void ClLogits1DNormKernel::configure(const CLCompileContext &compile_context, const ITensorInfo &src, const ITensorInfo &sum, ITensorInfo &dst, const SoftmaxKernelInfo &info)
{
    auto padding_info = get_padding_info({ &src, &dst, &sum });

    // Note: output should always have a scale of 1/256 and offset 0
    const bool                    is_quantized_asymmetric   = is_data_type_quantized_asymmetric(info.input_data_type);
    const DataType                output_data_type          = info.input_data_type;
    const QuantizationInfo        allowed_quantization_info = get_softmax_output_quantization_info(info.input_data_type, info.is_log);
    const UniformQuantizationInfo qinfo                     = src.quantization_info().uniform();

    // Output auto initialization if not yet initialized
    auto_init_if_empty(dst, src.clone()->set_data_type(output_data_type).set_quantization_info(allowed_quantization_info));

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_1DNorm(src, sum, dst, info));

    const auto         is_signed_qasymm8 = is_data_type_quantized_asymmetric_signed(info.input_data_type);
    const int          min_value         = is_signed_qasymm8 ? CL_SCHAR_MIN : 0;
    const unsigned int vector_size       = adjust_vec_size(16, src.dimension(0));

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(info.input_data_type));
    build_opts.add_option("-DMIN_VALUE=" + support::cpp11::to_string(min_value));
    build_opts.add_option("-DVECTOR_SIZE=" + support::cpp11::to_string(vector_size));
    build_opts.add_option("-DVECTOR_SIZE_LEFTOVER=" + support::cpp11::to_string(src.dimension(0) % vector_size));
    build_opts.add_option_if(is_data_type_quantized_asymmetric_signed(info.input_data_type), "-DQASYMM8_SIGNED");
    build_opts.add_options_if(is_quantized_asymmetric,
                              prepare_quantized_softmax_build_options(qinfo.scale, info.beta).options());
    build_opts.add_option_if(info.is_log, "-DLOG_SOFTMAX");

    // Create kernel
    std::string kernel_name = std::string("softmax_layer_norm") + (is_quantized_asymmetric ? "_quantized" : "");
    _kernel                 = create_kernel(compile_context, kernel_name, build_opts.options());

    // Configure window
    auto win = calculate_max_window(src, Steps(vector_size));
    ICLKernel::configure_internal(win);

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status ClLogits1DNormKernel::validate(const ITensorInfo &src, const ITensorInfo &sum, const ITensorInfo &dst, const SoftmaxKernelInfo &info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_1DNorm(src, sum, dst, info));

    return Status{};
}

void ClLogits1DNormKernel::run_op(ITensorPack &tensors, const Window &window, ::cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    auto src = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    auto dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));
    auto sum = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_INT_0));

    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst, sum);

    Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice            = window_collapsed.first_slice_window_3D();

    do
    {
        Window sum_slice = slice;
        sum_slice.set(Window::DimX, Window::Dimension(0, 1, 1));

        unsigned int idx = 0;
        // Set inputs
        add_3D_tensor_argument(idx, src, slice);
        add_3D_tensor_argument(idx, sum, sum_slice);
        add_3D_tensor_argument(idx, dst, slice);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(window_collapsed.slide_window_slice_3D(slice));
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute