/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "src/gpu/cl/kernels/ClElementwiseKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "src/common/utils/Validate.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/Cast.h"
#include "support/StringSupport.h"
#include <map>

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
namespace
{
constexpr unsigned int vector_size_byte_opencl = 16;

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

std::string generate_id_for_tuning_common(const std::string &kernel_name, const ITensorInfo &src1, const ITensorInfo &dst)
{
    std::string config_id;
    // Set config_id for enabling LWS tuning
    config_id = kernel_name;
    config_id += "_";
    config_id += lower_string(string_from_data_type(src1.data_type()));
    config_id += "_";
    config_id += support::cpp11::to_string(dst.dimension(0));
    config_id += "_";
    config_id += support::cpp11::to_string(dst.dimension(1));
    return config_id;
}

Status validate_in_place_output_shape(const bool in_place, const bool src1_in_place, const ITensorInfo &src1, const ITensorInfo &src2, const ITensorInfo &dst, const TensorShape &out_shape)
{
    if(in_place)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(out_shape, src1_in_place ? src1.tensor_shape() : src2.tensor_shape(), 0),
                                        "Wrong shape for dst, cannot do in_place calculation");
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(out_shape, dst.tensor_shape(), 0),
                                        "Wrong shape for dst");
    }
    return Status{};
}

Status validate_arguments_with_float_only_supported_rules(const ITensorInfo &src1, const ITensorInfo &src2, const ITensorInfo &dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(&src1, &src2, &dst);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(&src1);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src1, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src1, &src2);

    // Check whether it is in_place calculation
    const bool in_place      = (&src1 == &dst) || (&src2 == &dst);
    const bool src1_in_place = in_place && (&src1 == &dst);

    const TensorShape out_shape = TensorShape::broadcast_shape(src1.tensor_shape(), src2.tensor_shape());

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");

    // Validate in case of configured dst
    if(dst.total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&dst, 1, DataType::F16, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src1, &dst);
        ARM_COMPUTE_RETURN_ON_ERROR(validate_in_place_output_shape(in_place, src1_in_place, src1, src2, dst, out_shape));
    }

    return Status{};
}

Status validate_arguments_divide_operation(const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src1, src2, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src1);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src1, 1, DataType::F16, DataType::F32, DataType::S32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src1, src2);

    // Check whether it is in_place calculation
    const bool in_place      = (src1 == dst) || (src2 == dst);
    const bool src1_in_place = in_place && (src1 == dst);

    const TensorShape out_shape = TensorShape::broadcast_shape(src1->tensor_shape(), src2->tensor_shape());

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");

    // Validate in case of configured dst
    if(dst->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, DataType::F16, DataType::F32, DataType::S32);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src1, dst);
        ARM_COMPUTE_RETURN_ON_ERROR(validate_in_place_output_shape(in_place, src1_in_place, *src1, *src2, *dst, out_shape));
    }

    return Status{};
}

Status validate_arguments_with_arithmetic_rules(const ITensorInfo &src1, const ITensorInfo &src2, const ITensorInfo &dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(&src1);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src1, 1, DataType::U8, DataType::QASYMM8, DataType::QASYMM8_SIGNED,
                                                         DataType::S16, DataType::QSYMM16, DataType::F16,
                                                         DataType::S32, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src1, &src2);

    if(is_data_type_quantized_symmetric(src1.data_type()))
    {
        const int32_t in1_offset = src1.quantization_info().uniform().offset;
        const int32_t in2_offset = src2.quantization_info().uniform().offset;
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(in1_offset != 0, "For quantized symmetric, offset must be zero");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(in2_offset != 0, "For quantized symmetric, offset must be zero");
    }

    // Check whether it is in_place calculation
    const bool in_place      = (&src1 == &dst) || (&src2 == &dst);
    const bool src1_in_place = in_place && (&src1 == &dst);

    const TensorShape out_shape = TensorShape::broadcast_shape(src1.tensor_shape(), src2.tensor_shape());
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");

    // Validate in case of configured dst
    if(dst.total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src1, &dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(out_shape, dst.tensor_shape(), 0), "Wrong shape for dst");
        ARM_COMPUTE_RETURN_ON_ERROR(validate_in_place_output_shape(in_place, src1_in_place, src1, src2, dst, out_shape));

        if(is_data_type_quantized_symmetric(dst.data_type()))
        {
            const int32_t offset = dst.quantization_info().uniform().offset;
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(offset != 0, "For quantized symmetric, offset must be zero");
        }
    }
    return Status{};
}

CLBuildOptions generate_build_options_with_arithmetic_rules(const ITensorInfo &src1, const ITensorInfo &src2, const ITensorInfo &dst, const std::string &operation_string)
{
    CLBuildOptions build_opts;

    const unsigned int num_elems_processed_per_iteration = adjust_vec_size(vector_size_byte_opencl / dst.element_size(), dst.dimension(0));

    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(src1.data_type()));
    build_opts.add_option("-DVEC_SIZE_IN1=" + support::cpp11::to_string(src1.dimension(0) == 1 ? 1 : num_elems_processed_per_iteration));
    build_opts.add_option("-DVEC_SIZE_IN2=" + support::cpp11::to_string(src2.dimension(0) == 1 ? 1 : num_elems_processed_per_iteration));
    build_opts.add_option("-DVEC_SIZE_OUT=" + support::cpp11::to_string(num_elems_processed_per_iteration));
    build_opts.add_option("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(dst.dimension(0) % num_elems_processed_per_iteration));
    build_opts.add_option("-DOP=" + operation_string);
    if(is_data_type_quantized(src1.data_type()))
    {
        const UniformQuantizationInfo iq1info = src1.quantization_info().uniform();
        const UniformQuantizationInfo iq2info = src2.quantization_info().uniform();
        const UniformQuantizationInfo oqinfo  = dst.quantization_info().uniform();

        build_opts.add_option("-DOFFSET_IN1=" + support::cpp11::to_string(iq1info.offset));
        build_opts.add_option("-DOFFSET_IN2=" + support::cpp11::to_string(iq2info.offset));
        build_opts.add_option("-DOFFSET_OUT=" + support::cpp11::to_string(oqinfo.offset));
        build_opts.add_option("-DSCALE_IN1=" + float_to_string_with_full_precision(iq1info.scale));
        build_opts.add_option("-DSCALE_IN2=" + float_to_string_with_full_precision(iq2info.scale));
        build_opts.add_option("-DSCALE_OUT=" + float_to_string_with_full_precision(oqinfo.scale));
    }
    build_opts.add_option_if(src1.data_type() == DataType::S32, "-DS32");

    // Check whether it is in_place calculation
    const bool in_place      = (&src1 == &dst) || (&src2 == &dst);
    const bool src1_in_place = in_place && (&src1 == &dst);
    build_opts.add_option_if(in_place, "-DIN_PLACE");
    build_opts.add_option_if(src1_in_place, "-DSRC1_IN_PLACE");

    return build_opts;
}

std::pair<Status, Window> configure_window_arithmetic_common(ITensorInfo &dst)
{
    const unsigned int num_elems_processed_per_iteration = adjust_vec_size(vector_size_byte_opencl / dst.element_size(), dst.dimension(0));
    Window             win                               = calculate_max_window(dst, Steps(num_elems_processed_per_iteration));
    return std::make_pair(Status{}, win);
}

std::pair<Status, Window> validate_and_configure_window_for_arithmetic_operators(ITensorInfo &src1, ITensorInfo &src2, ITensorInfo &dst)
{
    const std::pair<TensorShape, ValidRegion> broadcast_pair = ITensorInfo::broadcast_shape_and_valid_region(src1, src2);
    const TensorShape &out_shape = broadcast_pair.first;

    auto_init_if_empty(dst, out_shape, 1, src1.data_type());

    return configure_window_arithmetic_common(dst);
}

std::pair<Status, Window> validate_and_configure_window_for_logical_binary_operators(ITensorInfo &src1, ITensorInfo &src2, ITensorInfo &dst)
{
    const std::pair<TensorShape, ValidRegion> broadcast_pair = ITensorInfo::broadcast_shape_and_valid_region(src1, src2);
    const TensorShape &out_shape = broadcast_pair.first;

    set_shape_if_empty(dst, out_shape);
    set_data_type_if_unknown(dst, DataType::U8);

    return configure_window_arithmetic_common(dst);
}

std::pair<Status, Window> validate_and_configure_window_for_division(ITensorInfo &src1, ITensorInfo &src2, ITensorInfo &dst)
{
    const std::pair<TensorShape, ValidRegion> broadcast_pair = ITensorInfo::broadcast_shape_and_valid_region(src1, src2);
    const TensorShape &out_shape = broadcast_pair.first;

    auto_init_if_empty(dst, out_shape, 1, src1.data_type());

    return configure_window_arithmetic_common(dst);
}
} // namespace

ClElementwiseKernel::ClElementwiseKernel()
{
    _type = CLKernelType::ELEMENTWISE;
}

void ClElementwiseKernel::configure_common(const ClCompileContext &compile_context, ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst)
{
    // Configure kernel window
    auto win_config = validate_and_configure_window(*src1, *src2, *dst);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);

    std::string kernel_name = "elementwise_operation_" + name();
    if(is_data_type_quantized(src1->data_type()))
    {
        kernel_name += "_quantized";
    }

    // Set kernel build options
    CLBuildOptions build_opts = generate_build_options(*src1, *src2, *dst);
    if(_act_info.enabled())
    {
        build_opts.add_option("-DACTIVATION_TYPE=" + lower_string(string_from_activation_func(_act_info.activation())));
        build_opts.add_option("-DA_VAL=" + float_to_string_with_full_precision(_act_info.a()));
        build_opts.add_option("-DB_VAL=" + float_to_string_with_full_precision(_act_info.b()));
    }

    // Create kernel
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    ICLKernel::configure_internal(win_config.second);

    _config_id = generate_id_for_tuning(kernel_name, *src1, *dst);
}

void ClElementwiseKernel::run_op(ITensorPack &tensors, const Window &window, ::cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const auto src_0 = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_0));
    const auto src_1 = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_1));
    auto       dst   = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    ARM_COMPUTE_ERROR_ON_NULLPTR(src_0, src_1, dst);

    const TensorShape &in_shape1 = src_0->info()->tensor_shape();
    const TensorShape &in_shape2 = src_1->info()->tensor_shape();
    const TensorShape &out_shape = dst->info()->tensor_shape();

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

    Window slice      = collapsed.first_slice_window_3D();
    Window slice_src1 = slice.broadcast_if_dimension_le_one(in_shape1_collapsed);
    Window slice_src2 = slice.broadcast_if_dimension_le_one(in_shape2_collapsed);

    // Check whether it is in_place calculation
    const bool in_place = (src_0 == dst) || (src_1 == dst);
    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, src_0, slice_src1);
        add_3D_tensor_argument(idx, src_1, slice_src2);
        if(!in_place)
        {
            add_3D_tensor_argument(idx, dst, slice);
        }

        enqueue(queue, *this, slice, lws_hint());
        ARM_COMPUTE_UNUSED(collapsed.slide_window_slice_3D(slice_src1));
        ARM_COMPUTE_UNUSED(collapsed.slide_window_slice_3D(slice_src2));
    }
    while(collapsed.slide_window_slice_3D(slice));
}

/** Logical binary */

void ClLogicalBinaryKernel::configure(const ClCompileContext &compile_context, LogicalOperation op, ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src1, src2, dst);
    ARM_COMPUTE_ERROR_THROW_ON(ClLogicalBinaryKernel::validate(op, src1, src2, dst));
    _op = op;
    configure_common(compile_context, src1, src2, dst);
}

Status ClLogicalBinaryKernel::validate(LogicalOperation op, const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst)
{
    ARM_COMPUTE_UNUSED(op);
    ARM_COMPUTE_ASSERT(op != LogicalOperation::Unknown && op != LogicalOperation::Not);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src1, src2, dst);

    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src1, 1, DataType::U8);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src1, src2);

    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_with_arithmetic_rules(*src1, *src2, *dst));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_for_logical_binary_operators(*src1->clone(), *src2->clone(), *dst->clone()).first);

    return Status{};
}

std::string ClLogicalBinaryKernel::name()
{
    switch(_op)
    {
        case LogicalOperation::And:
            return "AND";
        case LogicalOperation::Or:
            return "OR";
        case LogicalOperation::Not:
        /* fall through */
        default:
            ARM_COMPUTE_ASSERT(true);
    }
    return "";
}

std::pair<Status, Window> ClLogicalBinaryKernel::validate_and_configure_window(ITensorInfo &src1, ITensorInfo &src2, ITensorInfo &dst)
{
    return validate_and_configure_window_for_logical_binary_operators(src1, src2, dst);
}

CLBuildOptions ClLogicalBinaryKernel::generate_build_options(const ITensorInfo &src1, const ITensorInfo &src2, const ITensorInfo &dst)
{
    // The arithmetic utility functions can be share
    return generate_build_options_with_arithmetic_rules(src1, src2, dst, name());
}

std::string ClLogicalBinaryKernel::generate_id_for_tuning(const std::string &kernel_name, const ITensorInfo &src1, const ITensorInfo &dst)
{
    return generate_id_for_tuning_common(kernel_name, src1, dst);
}

/** Arithmetic operations with saturation*/
void ClSaturatedArithmeticKernel::configure(const ClCompileContext &compile_context, ArithmeticOperation op, ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output,
                                            const ConvertPolicy       &policy,
                                            const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, output);
    ARM_COMPUTE_ERROR_THROW_ON(ClSaturatedArithmeticKernel::validate(op, input1, input2, output, policy, act_info));
    auto padding_info = get_padding_info({ input1, input2, output });

    _policy   = policy;
    _op       = op;
    _act_info = act_info;
    configure_common(compile_context, input1, input2, output);
    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status ClSaturatedArithmeticKernel::validate(ArithmeticOperation op, const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ConvertPolicy &policy,
                                             const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(op, policy);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input1, input2, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_with_arithmetic_rules(*input1, *input2, *output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_for_arithmetic_operators(*input1->clone(), *input2->clone(), *output->clone()).first);
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.enabled() && !is_data_type_float(output->data_type()));

    return Status{};
}

std::pair<Status, Window> ClSaturatedArithmeticKernel::validate_and_configure_window(ITensorInfo &input1, ITensorInfo &input2, ITensorInfo &output)
{
    return validate_and_configure_window_for_arithmetic_operators(input1, input2, output);
}

CLBuildOptions ClSaturatedArithmeticKernel::generate_build_options(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output)
{
    const bool has_float_out = is_data_type_float(output.data_type());
    auto       build_options = generate_build_options_with_arithmetic_rules(input1, input2, output, name());
    build_options.add_option((_policy == ConvertPolicy::WRAP || has_float_out) ? "-DWRAP" : "-DSATURATE");
    return build_options;
}

std::string ClSaturatedArithmeticKernel::generate_id_for_tuning(const std::string &kernel_name, const ITensorInfo &input1, const ITensorInfo &output)
{
    auto config_id = generate_id_for_tuning_common(kernel_name, input1, output);
    config_id += (_policy == ConvertPolicy::WRAP) ? "_wrap_" : "_saturate_";
    config_id += lower_string(string_from_data_layout(input1.data_layout()));
    return config_id;
}

std::string ClSaturatedArithmeticKernel::name()
{
    return supported_sat_arithmetic_ops[_op];
}

/** Arithmetic operations*/
void ClArithmeticKernel::configure(const ClCompileContext &compile_context, ArithmeticOperation op, ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst,
                                   const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src1, src2, dst);
    ARM_COMPUTE_ERROR_THROW_ON(ClArithmeticKernel::validate(op, src1, src2, dst, act_info));
    auto padding_info = get_padding_info({ src1, src2, dst });

    _op       = op;
    _act_info = act_info;
    configure_common(compile_context, src1, src2, dst);
    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status ClArithmeticKernel::validate(ArithmeticOperation op, const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src1, src2, dst);
    if(op == ArithmeticOperation::DIV)
    {
        // Partial integer support S32/F32/F16
        ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_divide_operation(src1, src2, dst));
        ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_for_division(*src1->clone(), *src2->clone(), *dst->clone()).first);
    }
    else if(op == ArithmeticOperation::POWER)
    {
        // Power operators doesn't support integer arithmetic
        ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_with_float_only_supported_rules(*src1, *src2, *dst));
        ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_for_division(*src1->clone(), *src2->clone(), *dst->clone()).first);
    }
    else
    {
        ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_with_arithmetic_rules(*src1, *src2, *dst));
        ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_for_arithmetic_operators(*src1->clone(), *src2->clone(), *dst->clone()).first);
    }
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.enabled() && !is_data_type_float(dst->data_type()));

    return Status{};
}
std::pair<Status, Window> ClArithmeticKernel::validate_and_configure_window(ITensorInfo &src1, ITensorInfo &src2, ITensorInfo &dst)
{
    if(_op == ArithmeticOperation::DIV || _op == ArithmeticOperation::POWER)
    {
        // Division and Power operators don't support integer arithmetic
        return validate_and_configure_window_for_division(src1, src2, dst);
    }
    else
    {
        return validate_and_configure_window_for_arithmetic_operators(src1, src2, dst);
    }
}

CLBuildOptions ClArithmeticKernel::generate_build_options(const ITensorInfo &src1, const ITensorInfo &src2, const ITensorInfo &dst)
{
    return generate_build_options_with_arithmetic_rules(src1, src2, dst, name());
}
std::string ClArithmeticKernel::generate_id_for_tuning(const std::string &kernel_name, const ITensorInfo &src1, const ITensorInfo &dst)
{
    return generate_id_for_tuning_common(kernel_name, src1, dst);
}

std::string ClArithmeticKernel::name()
{
    return supported_arithmetic_ops[_op];
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
