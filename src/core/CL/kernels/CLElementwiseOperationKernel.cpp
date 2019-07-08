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
#include "arm_compute/core/CL/kernels/CLElementwiseOperationKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include <map>

namespace arm_compute
{
namespace
{
constexpr unsigned int num_elems_processed_per_iteration = 16;

std::map<ArithmeticOperation, std::string> supported_arithmetic_ops =
{
    { ArithmeticOperation::ADD, "ADD" },
    { ArithmeticOperation::SUB, "SUB" },
    { ArithmeticOperation::DIV, "DIV" },
    { ArithmeticOperation::SQUARED_DIFF, "SQUARED_DIFF" },
    { ArithmeticOperation::MIN, "MIN" },
    { ArithmeticOperation::MAX, "MAX" },
    { ArithmeticOperation::POWER, "POWER" },
    { ArithmeticOperation::PRELU, "PRELU" },
};

std::map<ArithmeticOperation, std::string> supported_sat_arithmetic_ops =
{
    { ArithmeticOperation::ADD, "ADD" },
    { ArithmeticOperation::SUB, "SUB" },
};

std::string generate_id_for_tuning_common(const std::string &kernel_name, const ITensorInfo &input1, const ITensorInfo &output)
{
    std::string config_id;
    // Set config_id for enabling LWS tuning
    config_id = kernel_name;
    config_id += "_";
    config_id += lower_string(string_from_data_type(input1.data_type()));
    config_id += "_";
    config_id += support::cpp11::to_string(output.dimension(0));
    config_id += "_";
    config_id += support::cpp11::to_string(output.dimension(1));
    return config_id;
}

Status validate_arguments_with_float_only_supported_rules(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(&input1, &input2, &output);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(&input1);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input1, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&input1, &input2);

    const TensorShape out_shape = TensorShape::broadcast_shape(input1.tensor_shape(), input2.tensor_shape());

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");

    // Validate in case of configured output
    if(output.total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&output, 1, DataType::F16, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&input1, &output);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(out_shape, output.tensor_shape(), 0),
                                        "Wrong shape for output");
    }

    return Status{};
}

Status validate_arguments_with_arithmetic_rules(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(&input1);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input1, 1, DataType::U8, DataType::QASYMM8, DataType::S16, DataType::QSYMM16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(&input2);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input2, 1, DataType::U8, DataType::QASYMM8, DataType::S16, DataType::QSYMM16, DataType::F16, DataType::F32);

    const bool is_quantized = is_data_type_quantized(input1.data_type()) || is_data_type_quantized(input2.data_type());
    if(is_quantized)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&input1, &input2);

        if(is_data_type_quantized_symmetric(input1.data_type()))
        {
            const int32_t in1_offset = input1.quantization_info().uniform().offset;
            const int32_t in2_offset = input2.quantization_info().uniform().offset;
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(in1_offset != 0, "For quantized symmetric, offset must be zero");
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(in2_offset != 0, "For quantized symmetric, offset must be zero");
        }
    }

    const TensorShape out_shape = TensorShape::broadcast_shape(input1.tensor_shape(), input2.tensor_shape());

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");

    // Validate in case of configured output
    if(output.total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(&output);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&output, 1, DataType::U8, DataType::QASYMM8, DataType::S16, DataType::QSYMM16, DataType::F16, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG((output.data_type() == DataType::U8) && ((input1.data_type() != DataType::U8) || (input2.data_type() != DataType::U8)),
                                        "Output can only be U8 if both inputs are U8");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(out_shape, output.tensor_shape(), 0),
                                        "Wrong shape for output");

        if(is_quantized)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&input1, &output);

            if(is_data_type_quantized_symmetric(output.data_type()))
            {
                const int32_t offset = output.quantization_info().uniform().offset;
                ARM_COMPUTE_RETURN_ERROR_ON_MSG(offset != 0, "For quantized symmetric, offset must be zero");
            }
        }
    }
    return Status{};
}

CLBuildOptions generate_build_options_with_arithmetic_rules(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output, const std::string &operation_string)
{
    CLBuildOptions build_opts;

    build_opts.add_option("-DDATA_TYPE_IN1=" + get_cl_type_from_data_type(input1.data_type()));
    build_opts.add_option("-DDATA_TYPE_IN2=" + get_cl_type_from_data_type(input2.data_type()));
    build_opts.add_option("-DDATA_TYPE_OUT=" + get_cl_type_from_data_type(output.data_type()));
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration));
    build_opts.add_option("-DOP=" + operation_string);
    if(is_data_type_quantized(input1.data_type()))
    {
        const UniformQuantizationInfo iq1info = input1.quantization_info().uniform();
        const UniformQuantizationInfo iq2info = input2.quantization_info().uniform();
        const UniformQuantizationInfo oqinfo  = output.quantization_info().uniform();

        build_opts.add_option("-DOFFSET_IN1=" + support::cpp11::to_string(iq1info.offset));
        build_opts.add_option("-DOFFSET_IN2=" + support::cpp11::to_string(iq2info.offset));
        build_opts.add_option("-DOFFSET_OUT=" + support::cpp11::to_string(oqinfo.offset));
        build_opts.add_option("-DSCALE_IN1=" + float_to_string_with_full_precision(iq1info.scale));
        build_opts.add_option("-DSCALE_IN2=" + float_to_string_with_full_precision(iq2info.scale));
        build_opts.add_option("-DSCALE_OUT=" + float_to_string_with_full_precision(oqinfo.scale));
    }
    return build_opts;
}

std::pair<Status, Window> configure_window_arithmetic_common(const ValidRegion &valid_region, ITensorInfo &input1, ITensorInfo &input2, ITensorInfo &output)
{
    Window win        = calculate_max_window(valid_region, Steps(num_elems_processed_per_iteration));
    Window win_input1 = win.broadcast_if_dimension_le_one(input1);
    Window win_input2 = win.broadcast_if_dimension_le_one(input2);

    AccessWindowHorizontal input1_access(&input1, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal input2_access(&input2, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(&output, 0, num_elems_processed_per_iteration);

    bool window_changed = update_window_and_padding(win_input1, input1_access)
                          || update_window_and_padding(win_input2, input2_access)
                          || update_window_and_padding(win, output_access);

    output_access.set_valid_region(win, valid_region);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

std::pair<Status, Window> validate_and_configure_window_for_arithmetic_operators(ITensorInfo &input1, ITensorInfo &input2, ITensorInfo &output)
{
    const std::pair<TensorShape, ValidRegion> broadcast_pair = ITensorInfo::broadcast_shape_and_valid_region(input1, input2);
    const TensorShape &out_shape    = broadcast_pair.first;
    const ValidRegion &valid_region = broadcast_pair.second;

    set_shape_if_empty(output, out_shape);

    if(input1.data_type() == DataType::S16 || input2.data_type() == DataType::S16)
    {
        set_format_if_unknown(output, Format::S16);
    }
    else if(input1.data_type() == DataType::F16 && input2.data_type() == DataType::F16)
    {
        set_format_if_unknown(output, Format::F16);
    }
    else if(input1.data_type() == DataType::F32 || input2.data_type() == DataType::F32)
    {
        set_format_if_unknown(output, Format::F32);
    }
    else if(input1.data_type() == DataType::QASYMM8 || input2.data_type() == DataType::QASYMM8)
    {
        set_data_type_if_unknown(output, DataType::QASYMM8);
    }
    else if(input1.data_type() == DataType::QSYMM16 || input2.data_type() == DataType::QSYMM16)
    {
        set_data_type_if_unknown(output, DataType::QSYMM16);
    }

    return configure_window_arithmetic_common(valid_region, input1, input2, output);
}

std::pair<Status, Window> validate_and_configure_window_for_division(ITensorInfo &input1, ITensorInfo &input2, ITensorInfo &output)
{
    const std::pair<TensorShape, ValidRegion> broadcast_pair = ITensorInfo::broadcast_shape_and_valid_region(input1, input2);
    const TensorShape &out_shape    = broadcast_pair.first;
    const ValidRegion &valid_region = broadcast_pair.second;
    auto_init_if_empty(output, out_shape, 1, input1.data_type());
    return configure_window_arithmetic_common(valid_region, input1, input2, output);
}
} // namespace

CLElementwiseOperationKernel::CLElementwiseOperationKernel()
    : _input1(nullptr), _input2(nullptr), _output(nullptr)
{
}

void CLElementwiseOperationKernel::configure_common(const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*input1->info(), *input2->info(), *output->info()));

    // Configure kernel window
    auto win_config = validate_and_configure_window(*input1->info(), *input2->info(), *output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);

    _input1 = input1;
    _input2 = input2;
    _output = output;

    std::string kernel_name = "elementwise_operation_" + name();
    if(is_data_type_quantized(input1->info()->data_type()))
    {
        kernel_name += "_quantized";
    }

    // Set kernel build options
    CLBuildOptions build_opts = generate_build_options(*input1->info(), *input2->info(), *output->info());

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));

    ICLKernel::configure_internal(win_config.second);

    _config_id = generate_id_for_tuning(kernel_name, *input1->info(), *output->info());
}

void CLElementwiseOperationKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const TensorShape &in_shape1 = _input1->info()->tensor_shape();
    const TensorShape &in_shape2 = _input2->info()->tensor_shape();
    const TensorShape &out_shape = _output->info()->tensor_shape();

    bool       can_collapse = true;
    const bool is_vector    = in_shape1.num_dimensions() == 1 || in_shape2.num_dimensions() == 1;
    if(std::min(in_shape1.total_size(), in_shape2.total_size()) > 1 && !is_vector)
    {
        can_collapse = (std::min(in_shape1.num_dimensions(), in_shape2.num_dimensions()) > Window::DimZ);
        for(size_t d = Window::DimZ; can_collapse && (d < out_shape.num_dimensions()); d++)
        {
            can_collapse = (in_shape1[d] == in_shape2[d]);
        }
    }

    bool   has_collapsed = false;
    Window collapsed     = can_collapse ? window.collapse_if_possible(ICLKernel::window(), Window::DimZ, &has_collapsed) : window;

    const TensorShape &in_shape1_collapsed = has_collapsed ? in_shape1.collapsed_from(Window::DimZ) : in_shape1;
    const TensorShape &in_shape2_collapsed = has_collapsed ? in_shape2.collapsed_from(Window::DimZ) : in_shape2;

    Window slice        = collapsed.first_slice_window_3D();
    Window slice_input1 = slice.broadcast_if_dimension_le_one(in_shape1_collapsed);
    Window slice_input2 = slice.broadcast_if_dimension_le_one(in_shape2_collapsed);

    do
    {
        unsigned int idx = 0;

        add_3D_tensor_argument(idx, _input1, slice_input1);
        add_3D_tensor_argument(idx, _input2, slice_input2);
        add_3D_tensor_argument(idx, _output, slice);

        enqueue(queue, *this, slice, lws_hint());

        ARM_COMPUTE_UNUSED(collapsed.slide_window_slice_3D(slice_input1));
        ARM_COMPUTE_UNUSED(collapsed.slide_window_slice_3D(slice_input2));
    }
    while(collapsed.slide_window_slice_3D(slice));
}

BorderSize CLElementwiseOperationKernel::border_size() const
{
    const unsigned int replicateSize = _output->info()->dimension(0) - std::min(_input1->info()->dimension(0), _input2->info()->dimension(0));
    const unsigned int border        = std::min<unsigned int>(num_elems_processed_per_iteration - 1U, replicateSize);
    return BorderSize{ 0, border, 0, 0 };
}

/** Arithmetic operations with saturation*/

void CLSaturatedArithmeticOperationKernel::configure(ArithmeticOperation op, const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, const ConvertPolicy &policy)
{
    _policy = policy;
    _op     = op;
    configure_common(input1, input2, output);
}

Status CLSaturatedArithmeticOperationKernel::validate(ArithmeticOperation op, const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ConvertPolicy &policy)
{
    ARM_COMPUTE_UNUSED(op, policy);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input1, input2, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_with_arithmetic_rules(*input1, *input2, *output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_for_arithmetic_operators(*input1->clone(), *input2->clone(), *output->clone()).first);

    return Status{};
}

std::pair<Status, Window> CLSaturatedArithmeticOperationKernel::validate_and_configure_window(ITensorInfo &input1, ITensorInfo &input2, ITensorInfo &output)
{
    return validate_and_configure_window_for_arithmetic_operators(input1, input2, output);
}

Status CLSaturatedArithmeticOperationKernel::validate_arguments(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output)
{
    return validate_arguments_with_arithmetic_rules(input1, input2, output);
}

CLBuildOptions CLSaturatedArithmeticOperationKernel::generate_build_options(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output)
{
    const bool has_float_out = is_data_type_float(output.data_type());
    auto       build_options = generate_build_options_with_arithmetic_rules(input1, input2, output, name());
    build_options.add_option((_policy == ConvertPolicy::WRAP || has_float_out) ? "-DWRAP" : "-DSATURATE");
    return build_options;
}
std::string CLSaturatedArithmeticOperationKernel::generate_id_for_tuning(const std::string &kernel_name, const ITensorInfo &input1, const ITensorInfo &output)
{
    auto config_id = generate_id_for_tuning_common(kernel_name, input1, output);
    config_id += (_policy == ConvertPolicy::WRAP) ? "_wrap_" : "_saturate_";
    config_id += lower_string(string_from_data_layout(input1.data_layout()));
    return config_id;
}

std::string CLSaturatedArithmeticOperationKernel::name()
{
    return supported_sat_arithmetic_ops[_op];
}

/** Arithmetic operations*/

void CLArithmeticOperationKernel::configure(ArithmeticOperation op, const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output)
{
    _op = op;
    configure_common(input1, input2, output);
}

Status CLArithmeticOperationKernel::validate(ArithmeticOperation op, const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input1, input2, output);
    if(op == ArithmeticOperation::DIV || op == ArithmeticOperation::POWER)
    {
        // Division and Power operators don't support integer arithmetic
        ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_with_float_only_supported_rules(*input1, *input2, *output));
        ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_for_division(*input1->clone(), *input2->clone(), *output->clone()).first);
    }
    else
    {
        ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_with_arithmetic_rules(*input1, *input2, *output));
        ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_for_arithmetic_operators(*input1->clone(), *input2->clone(), *output->clone()).first);
    }

    return Status{};
}
std::pair<Status, Window> CLArithmeticOperationKernel::validate_and_configure_window(ITensorInfo &input1, ITensorInfo &input2, ITensorInfo &output)
{
    if(_op == ArithmeticOperation::DIV || _op == ArithmeticOperation::POWER)
    {
        // Division and Power operators don't support integer arithmetic
        return validate_and_configure_window_for_division(input1, input2, output);
    }
    else
    {
        return validate_and_configure_window_for_arithmetic_operators(input1, input2, output);
    }
}
Status CLArithmeticOperationKernel::validate_arguments(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output)
{
    if(_op == ArithmeticOperation::DIV || _op == ArithmeticOperation::POWER)
    {
        // Division and Power operators don't support integer arithmetic
        return validate_arguments_with_float_only_supported_rules(input1, input2, output);
    }
    else
    {
        return validate_arguments_with_arithmetic_rules(input1, input2, output);
    }
}

CLBuildOptions CLArithmeticOperationKernel::generate_build_options(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output)
{
    return generate_build_options_with_arithmetic_rules(input1, input2, output, name());
}
std::string CLArithmeticOperationKernel::generate_id_for_tuning(const std::string &kernel_name, const ITensorInfo &input1, const ITensorInfo &output)
{
    return generate_id_for_tuning_common(kernel_name, input1, output);
}

std::string CLArithmeticOperationKernel::name()
{
    return supported_arithmetic_ops[_op];
}
} // namespace arm_compute
