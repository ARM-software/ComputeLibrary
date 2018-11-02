/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLSoftmaxLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"

#include <set>
#include <string>

using namespace arm_compute;

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
                                       (1ll << 31) - 1.0);
    int input_beta_multiplier, input_beta_left_shift;
    quantization::calculate_quantized_multiplier_greater_than_one(beta_multiplier, &input_beta_multiplier, &input_beta_left_shift);

    const double max_input_rescaled = 1.0 * ((1 << scaled_diff_int_bits) - 1) * (1ll << (31 - scaled_diff_int_bits)) / (1ll << input_beta_left_shift);
    const int    diff_min           = -1.f * std::floor(max_input_rescaled);

    CLBuildOptions build_opts;
    build_opts.add_option("-DSCALED_DIFF_INT_BITS=" + support::cpp11::to_string(scaled_diff_int_bits));
    build_opts.add_option("-DEXP_ACCUMULATION_INT_BITS=" + support::cpp11::to_string(exp_accumulation_in_bits));
    build_opts.add_option("-DINPUT_BETA_MULTIPLIER=" + support::cpp11::to_string(input_beta_multiplier));
    build_opts.add_option("-DINPUT_BETA_LEFT_SHIFT=" + support::cpp11::to_string(input_beta_left_shift));
    build_opts.add_option("-DDIFF_MIN=" + support::cpp11::to_string(diff_min));

    return build_opts;
}

// Arguments Validation

Status validate_arguments_1DMax(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QASYMM8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        // Softmax across the x dimension
        TensorShape output_shape{ input->tensor_shape() };
        output_shape.set(0, 1);

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT_POSITION(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), output_shape);
    }

    return Status{};
}

Status validate_arguments_1DShiftExpSum(const ITensorInfo *input, const ITensorInfo *max, const ITensorInfo *output, const ITensorInfo *sum)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QASYMM8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(max, sum, output);

    const bool is_quantized_asymmetric = is_data_type_quantized_asymmetric(input->data_type());

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        if(is_quantized_asymmetric)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::S32);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        }
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT_POSITION(input, output);
    }

    // Checks performed when sum is configured
    if(sum->total_size() != 0)
    {
        if(is_quantized_asymmetric)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(sum, 1, DataType::S32);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(max, sum);
        }
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(max, sum);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT_POSITION(max, sum);
    }

    return Status{};
}

Status validate_arguments_1DMaxShiftExpSum(const ITensorInfo *input, const ITensorInfo *max, const ITensorInfo *output, const ITensorInfo *sum)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(max, sum, output);

    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, max);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT_POSITION(input, max);

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT_POSITION(input, output);
    }

    // Checks performed when sum is configured
    if(sum->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(max, sum);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(max, sum);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT_POSITION(max, sum);
    }

    return Status{};
}

Status validate_arguments_1DNorm(const ITensorInfo *input, const ITensorInfo *sum, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QS16, DataType::S32, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(sum, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, sum);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT_POSITION(input, sum);

    // Note: output should always have a scale of 1/256 and offset 0
    const QuantizationInfo allowed_quantization_info = QuantizationInfo(1.f / 256, 0);
    const bool             is_quantized_asymmetric   = (input->data_type() == DataType::S32);

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT_POSITION(input, output);
        if(!is_quantized_asymmetric)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::QASYMM8);
            ARM_COMPUTE_RETURN_ERROR_ON(output->quantization_info() != allowed_quantization_info);
        }
    }

    return Status{};
}

// Window validation

std::pair<Status, Window> validate_and_configure_window_1DMax(ITensorInfo *input, ITensorInfo *output)
{
    TensorShape output_shape{ input->tensor_shape() };
    output_shape.set(0, 1);

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(output_shape));

    // The kernel loops over all elements in steps of 16
    const unsigned int     num_elems_processed_per_iteration = ceil_to_multiple(input->dimension(0), 16);
    constexpr unsigned int num_elems_written_per_iteration   = 1;

    Window                 win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output, 0, num_elems_written_per_iteration);

    bool window_changed = update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

std::pair<Status, Window> validate_and_configure_window_1DShiftExpSum(ITensorInfo *input, ITensorInfo *max, ITensorInfo *output, ITensorInfo *sum)
{
    const bool     is_quantized_asymmetric = is_data_type_quantized_asymmetric(input->data_type());
    const DataType tmp_data_type           = is_quantized_asymmetric ? DataType::S32 : input->data_type();

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*sum, max->clone()->set_data_type(tmp_data_type).set_fixed_point_position(input->fixed_point_position()));
    auto_init_if_empty(*output, input->clone()->set_data_type(tmp_data_type));

    // The kernel loops over all elements in steps of 16
    const unsigned int num_elems_processed_per_iteration = ceil_to_multiple(input->dimension(0), 16);

    Window win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal max_access(max, 0, 1);
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal sum_access(sum, 0, 1);

    bool window_changed = update_window_and_padding(win, input_access, max_access, output_access, sum_access);

    output_access.set_valid_region(win, input->valid_region());
    sum_access.set_valid_region(win, ValidRegion(Coordinates(), sum->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

std::pair<Status, Window> validate_and_configure_window_1DMaxShiftExpSum(ITensorInfo *input, ITensorInfo *max, ITensorInfo *output, ITensorInfo *sum)
{
    // Output auto initialization if not yet initialized
    auto_init_if_empty(*sum, input->clone()->set_tensor_shape(max->tensor_shape()));
    auto_init_if_empty(*output, *input->clone());

    CLLogits1DMaxShiftExpSumKernel::ParallelReductionInfo parallel_reduction_info = CLLogits1DMaxShiftExpSumKernel::is_parallel_reduction(input->dimension(0));
    unsigned int                                          vector_size             = std::get<1>(parallel_reduction_info);
    const unsigned int                                    num_elems_x             = ceil_to_multiple(input->tensor_shape().x(), vector_size);
    Window                                                win                     = calculate_max_window(*input, Steps(num_elems_x));

    AccessWindowHorizontal input_access(input, 0, num_elems_x);
    AccessWindowHorizontal max_access(max, 0, 1);
    AccessWindowHorizontal output_access(output, 0, num_elems_x);
    AccessWindowHorizontal sum_access(sum, 0, 1);

    bool window_changed = update_window_and_padding(win, input_access, max_access, output_access, sum_access);

    output_access.set_valid_region(win, input->valid_region());
    sum_access.set_valid_region(win, ValidRegion(Coordinates(), sum->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

std::pair<Status, Window> validate_and_configure_window_1DNorm(ITensorInfo *input, ITensorInfo *output, ITensorInfo *sum)
{
    const QuantizationInfo allowed_quantization_info = QuantizationInfo(1.f / 256, 0);
    const bool             is_quantized_asymmetric   = (input->data_type() == DataType::S32);
    const DataType         output_data_type          = is_quantized_asymmetric ? DataType::QASYMM8 : input->data_type();

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output,
                       input->clone()->set_data_type(output_data_type).set_quantization_info(allowed_quantization_info));

    constexpr unsigned int num_elems_processed_per_iteration = 16;

    Window win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
    AccessWindowStatic     sum_access(sum, 0, 0, 1, sum->dimension(1));
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);

    bool window_changed = update_window_and_padding(win, input_access, sum_access, output_access);

    output_access.set_valid_region(win, input->valid_region());

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

} // namespace

void CLLogits1DMaxKernel::configure(const ICLTensor *input, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    TensorShape output_shape{ input->info()->tensor_shape() };
    output_shape.set(0, 1);

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(output_shape));

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_1DMax(input->info(), output->info()));

    _input  = input;
    _output = output;

    const DataType data_type = input->info()->data_type();

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type));
    build_opts.add_option_if(is_data_type_fixed_point(data_type),
                             "-DFIXED_POINT_POSITION=" + support::cpp11::to_string(input->info()->fixed_point_position()));
    build_opts.add_option_if(data_type == DataType::F16, "-DUSE_F16");
    // Tell the kernel that the width is not a multiple of 16
    build_opts.add_option_if((input->info()->dimension(0) % max_cl_vector_width) != 0, "-DNON_MULTIPLE_OF_16");

    // Create kernel
    std::string kernel_name = is_data_type_quantized_asymmetric(data_type) ? "softmax_layer_max_quantized" : "softmax_layer_max";
    _kernel                 = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));

    // Set fixed arguments
    unsigned int idx = 2 * num_arguments_per_3D_tensor(); //Skip the input and output parameters
    _kernel.setArg<cl_uint>(idx++, input->info()->dimension(0));

    // Configure kernel window
    auto win_config = validate_and_configure_window_1DMax(input->info(), output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure(win_config.second);

    // Set config_id for enabling LWS tuning
    _config_id = "softmax_layer_";
    _config_id += lower_string(string_from_data_type(data_type));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(1));
}

Status CLLogits1DMaxKernel::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_1DMax(input, output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_1DMax(input->clone().get(), output->clone().get()).first);

    return Status{};
}

CLLogits1DShiftExpSumKernel::CLLogits1DShiftExpSumKernel()
    : _input(nullptr), _max(nullptr), _output(nullptr), _sum(nullptr)
{
}

void CLLogits1DShiftExpSumKernel::configure(const ICLTensor *input, const ICLTensor *max, ICLTensor *output, ICLTensor *sum, float beta)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, max, sum, output);

    const bool     is_quantized_asymmetric = is_data_type_quantized_asymmetric(input->info()->data_type());
    const DataType tmp_data_type           = is_quantized_asymmetric ? DataType::S32 : input->info()->data_type();

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*sum->info(), max->info()->clone()->set_data_type(tmp_data_type).set_fixed_point_position(input->info()->fixed_point_position()));
    auto_init_if_empty(*output->info(), input->info()->clone()->set_data_type(tmp_data_type));

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_1DShiftExpSum(input->info(), max->info(), output->info(), sum->info()));

    _input  = input;
    _max    = max;
    _output = output;
    _sum    = sum;

    const DataType dt       = input->info()->data_type();
    auto           beta_int = static_cast<int>(lround(beta * (1 << input->info()->fixed_point_position())));

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option(std::string("-DDATA_TYPE=" + get_cl_type_from_data_type(dt)));
    build_opts.add_option_if(is_data_type_fixed_point(dt),
                             std::string("-DFIXED_POINT_POSITION=" + support::cpp11::to_string(input->info()->fixed_point_position())));
    build_opts.add_option_if(dt == DataType::F16, std::string("-DUSE_F16"));
    // Tell the kernel that the width is not a multiple of 16
    build_opts.add_option_if((input->info()->dimension(0) % max_cl_vector_width) != 0, std::string("-DNON_MULTIPLE_OF_16"));
    build_opts.add_option_if(is_data_type_fixed_point(dt) && (beta != 1.0f), std::string("-DBETA=" + support::cpp11::to_string(beta_int)));
    build_opts.add_option_if(is_data_type_float(dt) && (beta != 1.0f), std::string("-DBETA=" + float_to_string_with_full_precision(beta)));
    build_opts.add_options_if(is_quantized_asymmetric,
                              prepare_quantized_softmax_build_options(input->info()->quantization_info().scale, beta).options());

    // Create kernel
    std::string kernel_name = is_quantized_asymmetric ? "softmax_layer_shift_exp_sum_quantized" : "softmax_layer_shift_exp_sum";
    _kernel                 = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));

    // Set fixed arguments
    unsigned int idx = 4 * num_arguments_per_3D_tensor(); //Skip the input and output parameters
    _kernel.setArg<cl_uint>(idx++, input->info()->dimension(0));

    // Configure window
    auto win_config = validate_and_configure_window_1DShiftExpSum(input->info(), max->info(), output->info(), sum->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure(win_config.second);
}

Status CLLogits1DShiftExpSumKernel::validate(const ITensorInfo *input, const ITensorInfo *max, const ITensorInfo *output, const ITensorInfo *sum)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_1DShiftExpSum(input, max, output, sum));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_1DShiftExpSum(input->clone().get(), max->clone().get(), output->clone().get(), sum->clone().get()).first);

    return Status{};
}

void CLLogits1DShiftExpSumKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice            = window_collapsed.first_slice_window_3D();

    do
    {
        unsigned int idx = 0;
        // Set inputs
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _max, slice);
        add_3D_tensor_argument(idx, _output, slice);
        add_3D_tensor_argument(idx, _sum, slice);
        enqueue(queue, *this, slice, _lws_hint);
    }
    while(window_collapsed.slide_window_slice_3D(slice));
}

/**< Grid size (obtained through auto-tuning) */
const unsigned int CLLogits1DMaxShiftExpSumKernel::_grid_size = 64;
/**< Vector size in the serial case (obtained through auto-tuning) */
const unsigned int CLLogits1DMaxShiftExpSumKernel::_serial_vector_size = 8;
/**< Vector size in the parallel case (obtained through auto-tuning, enables the best memory access pattern for Bifrost) .*/
const unsigned int CLLogits1DMaxShiftExpSumKernel::_parallel_vector_size = 4;

CLLogits1DMaxShiftExpSumKernel::CLLogits1DMaxShiftExpSumKernel()
    : _input(nullptr), _max(nullptr), _output(nullptr), _sum(nullptr)
{
}

void CLLogits1DMaxShiftExpSumKernel::configure(const ICLTensor *input, ICLTensor *max, ICLTensor *output, ICLTensor *sum, float beta)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, max, sum, output);

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*sum->info(), input->info()->clone()->set_tensor_shape(max->info()->tensor_shape()));
    auto_init_if_empty(*output->info(), *input->info()->clone());

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_1DMaxShiftExpSum(input->info(), max->info(), output->info(), sum->info()));

    _input  = input;
    _max    = max;
    _output = output;
    _sum    = sum;

    const DataType dt                 = input->info()->data_type();
    const size_t   reduction_dim_size = input->info()->dimension(0);
    auto           beta_int           = static_cast<int>(lround(beta * (1 << input->info()->fixed_point_position())));

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(dt));
    build_opts.add_option_if(is_data_type_fixed_point(dt),
                             "-DFIXED_POINT_POSITION=" + support::cpp11::to_string(input->info()->fixed_point_position()));
    build_opts.add_option_if(dt == DataType::F16, "-DUSE_F16");
    build_opts.add_option_if(is_data_type_fixed_point(dt) && (beta != 1.0f), "-DBETA=" + support::cpp11::to_string(beta_int));
    build_opts.add_option_if(is_data_type_float(dt) && (beta != 1.0f), "-DBETA=" + float_to_string_with_full_precision(beta));

    _lws_hint                                     = cl::NullRange;
    std::string           kernel_name             = std::string("softmax_layer_max_shift_exp_sum_serial");
    ParallelReductionInfo parallel_reduction_info = is_parallel_reduction(reduction_dim_size);
    unsigned int          vector_size             = std::get<1>(parallel_reduction_info);

    build_opts.add_option("-DVECTOR_SIZE=" + support::cpp11::to_string(vector_size));
    build_opts.add_option("-DLOG_VECTOR_SIZE=" + support::cpp11::to_string(lround(log2(vector_size))));
    build_opts.add_option_if((reduction_dim_size % vector_size) != 0, "-DNON_MULTIPLE_OF_VECTOR_SIZE");

    // Configure parallel kernel if needed
    if(std::get<0>(parallel_reduction_info))
    {
        kernel_name            = std::string("softmax_layer_max_shift_exp_sum_parallel");
        bool is_grid_size_pow2 = (_grid_size != 0) && ((_grid_size & (_grid_size - 1)) == 0);
        build_opts.add_option_if(is_grid_size_pow2 && _grid_size <= 256, "-DGRID_SIZE=" + support::cpp11::to_string(_grid_size));

        // Handle boundary conditions.
        const unsigned int multiple_grid_size = (reduction_dim_size / vector_size) % _grid_size;
        build_opts.add_option_if((multiple_grid_size != 0) || ((reduction_dim_size % vector_size) != 0), "-DNON_MULTIPLE_OF_GRID_SIZE");
        // Setting _lws_hint in this way can also communicate grid_size to CLLogits1DMaxShiftExpSumKernel::run().
        // A single workgroup performs reduction in dimension 0 in the parallel case, hence lws[0]==gws[0].
        _lws_hint = cl::NDRange(_grid_size);
    }

    // Create kernel.
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));

    // Set static arguments. Both the kernels use the same arguments
    unsigned int idx = 4 * num_arguments_per_3D_tensor(); //Skip the input and output parameters
    _kernel.setArg<cl_uint>(idx++, reduction_dim_size);

    // Configure window
    auto win_config = validate_and_configure_window_1DMaxShiftExpSum(input->info(), max->info(), output->info(), sum->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure(win_config.second);
}

Status CLLogits1DMaxShiftExpSumKernel::validate(const ITensorInfo *input, const ITensorInfo *max, const ITensorInfo *output, const ITensorInfo *sum)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_1DMaxShiftExpSum(input, max, output, sum));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_1DMaxShiftExpSum(input->clone().get(), max->clone().get(), output->clone().get(), sum->clone().get()).first);

    return Status{};
}

CLLogits1DMaxShiftExpSumKernel::ParallelReductionInfo CLLogits1DMaxShiftExpSumKernel::is_parallel_reduction(size_t size)
{
    bool         is_parallel_reduction = (size >= (_grid_size * _serial_vector_size)) && (_grid_size > 1);
    unsigned int vector_size           = is_parallel_reduction ? _parallel_vector_size : _serial_vector_size;
    return std::make_tuple(is_parallel_reduction, vector_size);
}

void CLLogits1DMaxShiftExpSumKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    // Collapse window in Z dimension
    Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);

    // Reconfigure window in case of parallel reduction
    ParallelReductionInfo parallel_reduction_info = is_parallel_reduction(_input->info()->dimension(0));
    if(std::get<0>(parallel_reduction_info))
    {
        // To launch grid_size parallel workitems, steps.x should be modified as follows.
        const unsigned int step = std::get<1>(parallel_reduction_info);
        window_collapsed.set(Window::DimX, Window::Dimension(0, _grid_size * step, step));
    }

    // Get slices
    Window slice = window_collapsed.first_slice_window_3D();
    do
    {
        unsigned int idx = 0;
        // Set inputs
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _max, slice);
        add_3D_tensor_argument(idx, _output, slice);
        add_3D_tensor_argument(idx, _sum, slice);
        enqueue(queue, *this, slice, _lws_hint);
    }
    while(window_collapsed.slide_window_slice_3D(slice));
}

CLLogits1DNormKernel::CLLogits1DNormKernel()
    : _input(nullptr), _sum(nullptr), _output(nullptr)
{
}

void CLLogits1DNormKernel::configure(const ICLTensor *input, const ICLTensor *sum, ICLTensor *output, float beta)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, sum, output);

    // Note: output should always have a scale of 1/256 and offset 0
    const QuantizationInfo allowed_quantization_info = QuantizationInfo(1.f / 256, 0);
    const bool             is_quantized_asymmetric   = (input->info()->data_type() == DataType::S32);
    const DataType         output_data_type          = is_quantized_asymmetric ? DataType::QASYMM8 : input->info()->data_type();

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output->info(),
                       input->info()->clone()->set_data_type(output_data_type).set_quantization_info(allowed_quantization_info));

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_1DNorm(input->info(), sum->info(), output->info()));

    _input  = input;
    _sum    = sum;
    _output = output;

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    build_opts.add_option_if(is_data_type_fixed_point(input->info()->data_type()),
                             "-DFIXED_POINT_POSITION=" + support::cpp11::to_string(input->info()->fixed_point_position()));
    build_opts.add_options_if(is_quantized_asymmetric,
                              prepare_quantized_softmax_build_options(input->info()->quantization_info().scale, beta).options());

    // Create kernel
    std::string kernel_name = is_quantized_asymmetric ? "softmax_layer_norm_quantized" : "softmax_layer_norm";
    _kernel                 = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));

    // Configure window
    auto win_config = validate_and_configure_window_1DNorm(input->info(), output->info(), sum->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure(win_config.second);
}

Status CLLogits1DNormKernel::validate(const ITensorInfo *input, const ITensorInfo *sum, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_1DNorm(input, sum, output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_1DNorm(input->clone().get(), output->clone().get(), sum->clone().get()).first);

    return Status{};
}

void CLLogits1DNormKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice            = window_collapsed.first_slice_window_3D();

    do
    {
        Window sum_slice = slice;
        sum_slice.set(Window::DimX, Window::Dimension(0, 1, 1));

        unsigned int idx = 0;
        // Set inputs
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _sum, sum_slice);
        add_3D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice, _lws_hint);
    }
    while(window_collapsed.slide_window_slice_3D(slice));
}
