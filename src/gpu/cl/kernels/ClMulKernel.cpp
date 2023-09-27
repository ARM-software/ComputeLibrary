/*
 * Copyright (c) 2016-2021, 2023 Arm Limited.
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
#include "src/gpu/cl/kernels/ClMulKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/ActivationFunctionUtils.h"
#include "arm_compute/core/utils/helpers/AdjustVecSize.h"
#include "arm_compute/core/utils/StringUtils.h"

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
Status validate_arguments(const ITensorInfo         *src1,
                          const ITensorInfo         *src2,
                          const ITensorInfo         *dst,
                          float                      scale,
                          ConvertPolicy              overflow_policy,
                          RoundingPolicy             rounding_policy,
                          const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(overflow_policy);
    ARM_COMPUTE_UNUSED(rounding_policy);

    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src1, src2, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src1);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src1, 1, DataType::U8, DataType::QASYMM8,
                                                         DataType::QASYMM8_SIGNED, DataType::S16, DataType::QSYMM16,
                                                         DataType::F16, DataType::S32, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src2, 1, DataType::U8, DataType::QASYMM8,
                                                         DataType::QASYMM8_SIGNED, DataType::S16, DataType::QSYMM16,
                                                         DataType::F16, DataType::S32, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(scale < 0, "Scale cannot be negative.");
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.enabled() && !is_data_type_float(dst->data_type()));

    // Check whether it is in_place calculation
    const bool in_place      = (src1 == dst) || (src2 == dst);
    const bool src1_in_place = in_place && (src1 == dst);

    const TensorShape &out_shape = TensorShape::broadcast_shape(src1->tensor_shape(), src2->tensor_shape());

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");

    // Validate in case of configured dst
    if (dst->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, DataType::U8, DataType::QASYMM8,
                                                             DataType::QASYMM8_SIGNED, DataType::S16, DataType::QSYMM16,
                                                             DataType::F16, DataType::S32, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(dst->data_type() == DataType::U8 &&
                                            (src1->data_type() != DataType::U8 || src2->data_type() != DataType::U8),
                                        "Dst can only be U8 if both src are U8");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(
            dst->data_type() == DataType::QASYMM8 &&
                (src1->data_type() != DataType::QASYMM8 || src2->data_type() != DataType::QASYMM8),
            "Dst can only be QASYMM8 if both src are QASYMM8");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(
            dst->data_type() == DataType::QASYMM8_SIGNED &&
                (src1->data_type() != DataType::QASYMM8_SIGNED || src2->data_type() != DataType::QASYMM8_SIGNED),
            "Dst can only be QASYMM8_SIGNED if both src are QASYMM8_SIGNED");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(
            dst->data_type() == DataType::QSYMM16 &&
                (src1->data_type() != DataType::QSYMM16 || src2->data_type() != DataType::QSYMM16),
            "Dst can only be QSYMM16 if both src are QSYMM16");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG((src1->data_type() == DataType::S32 || src2->data_type() == DataType::S32) &&
                                            (dst->data_type() != DataType::S32),
                                        "Dst must be S32 if source tensors are S32");
        if (in_place)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(
                detail::have_different_dimensions(out_shape,
                                                  src1_in_place ? src1->tensor_shape() : src2->tensor_shape(), 0),
                "Wrong shape for dst, cannot do in_place calculation");
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(out_shape, dst->tensor_shape(), 0),
                                            "Wrong shape for dst");
        }
    }

    return Status{};
}
} // namespace

ClMulKernel::ClMulKernel()
{
    _type = CLKernelType::ELEMENTWISE;
}

void ClMulKernel::configure(const CLCompileContext    &compile_context,
                            ITensorInfo               *src1,
                            ITensorInfo               *src2,
                            ITensorInfo               *dst,
                            float                      scale,
                            ConvertPolicy              overflow_policy,
                            RoundingPolicy             rounding_policy,
                            const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src1, src2, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src1, src2, dst, scale, overflow_policy, rounding_policy, act_info));

    auto padding_info = get_padding_info({src1, src2, dst});

    const TensorShape &out_shape = TensorShape::broadcast_shape(src1->tensor_shape(), src2->tensor_shape());
    auto_init_if_empty(*dst, src1->clone()->set_tensor_shape(out_shape));

    int scale_int = -1;
    // Extract sign, exponent and mantissa
    int   exponent            = 0;
    float normalized_mantissa = std::frexp(scale, &exponent);
    // Use int scaling if factor is equal to 1/2^n for 0 <= n <= 15
    // frexp returns 0.5 as mantissa which means that the exponent will be in the range of -1 <= e <= 14
    // Moreover, it will be negative as we deal with 1/2^n
    if ((normalized_mantissa == 0.5f) && (-14 <= exponent) && (exponent <= 1))
    {
        // Store the positive exponent. We know that we compute 1/2^n
        // Additionally we need to subtract 1 to compensate that frexp used a mantissa of 0.5
        scale_int = std::abs(exponent - 1);
    }

    std::string acc_type;
    // Check if it has float src and dst
    if (is_data_type_float(src1->data_type()) || is_data_type_float(src2->data_type()))
    {
        scale_int = -1;
        acc_type  = (src1->data_type() == DataType::F32 || src2->data_type() == DataType::F32) ? "float" : "half";
    }
    else
    {
        if (src1->element_size() == 4 || src2->element_size() == 4)
        {
            // use 64 bit accumulator for 32-bit input
            acc_type = "long";
        }
        else if (src1->element_size() == 2 || src2->element_size() == 2)
        {
            // Use 32-bit accumulator for 16-bit input
            acc_type = "int";
        }
        else
        {
            // Use 16-bit accumulator for 8-bit input
            acc_type = "ushort";
        }
    }

    const bool         is_quantized      = is_data_type_quantized(src1->data_type());
    const unsigned int vec_size          = adjust_vec_size(16 / dst->element_size(), dst->dimension(0));
    const unsigned int vec_size_leftover = dst->dimension(0) % vec_size;

    // Set kernel build options
    std::string    kernel_name = "pixelwise_mul";
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE_IN1=" + get_cl_type_from_data_type(src1->data_type()));
    build_opts.add_option("-DDATA_TYPE_IN2=" + get_cl_type_from_data_type(src2->data_type()));
    build_opts.add_option("-DDATA_TYPE_OUT=" + get_cl_type_from_data_type(dst->data_type()));
    build_opts.add_option("-DVEC_SIZE_IN1=" + ((dst->dimension(0) != 1 && src1->dimension(0) == 1)
                                                   ? "1"
                                                   : support::cpp11::to_string(vec_size)));
    build_opts.add_option("-DVEC_SIZE_IN2=" + ((dst->dimension(0) != 1 && src2->dimension(0) == 1)
                                                   ? "1"
                                                   : support::cpp11::to_string(vec_size)));
    build_opts.add_option("-DVEC_SIZE_OUT=" + support::cpp11::to_string(vec_size));
    build_opts.add_option("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(vec_size_leftover));
    if (is_quantized && (dst->data_type() != DataType::S32))
    {
        const UniformQuantizationInfo iq1_info = src1->quantization_info().uniform();
        const UniformQuantizationInfo iq2_info = src2->quantization_info().uniform();
        const UniformQuantizationInfo oq_info  = dst->quantization_info().uniform();

        build_opts.add_option_if(is_data_type_quantized_asymmetric(src1->data_type()),
                                 "-DOFFSET_IN1=" + support::cpp11::to_string(iq1_info.offset));
        build_opts.add_option_if(is_data_type_quantized_asymmetric(src2->data_type()),
                                 "-DOFFSET_IN2=" + support::cpp11::to_string(iq2_info.offset));
        build_opts.add_option_if(is_data_type_quantized_asymmetric(dst->data_type()),
                                 "-DOFFSET_OUT=" + support::cpp11::to_string(oq_info.offset));
        build_opts.add_option("-DSCALE_IN1=" + float_to_string_with_full_precision(iq1_info.scale));
        build_opts.add_option("-DSCALE_IN2=" + float_to_string_with_full_precision(iq2_info.scale));
        build_opts.add_option("-DSCALE_OUT=" + float_to_string_with_full_precision(oq_info.scale));
        kernel_name += "_quantized";
    }
    else
    {
        kernel_name += (scale_int >= 0) ? "_int" : "_float";
        build_opts.add_option_if_else(overflow_policy == ConvertPolicy::WRAP || is_data_type_float(dst->data_type()),
                                      "-DWRAP", "-DSATURATE");
        build_opts.add_option_if_else(rounding_policy == RoundingPolicy::TO_ZERO, "-DROUND=_rtz", "-DROUND=_rte");
        build_opts.add_option("-DACC_DATA_TYPE=" + acc_type);
        if (act_info.enabled())
        {
            build_opts.add_option("-DACTIVATION_TYPE=" +
                                  lower_string(string_from_activation_func(act_info.activation())));
            build_opts.add_option("-DA_VAL=" + float_to_string_with_full_precision(act_info.a()));
            build_opts.add_option("-DB_VAL=" + float_to_string_with_full_precision(act_info.b()));
        }
    }

    // Check whether it is in_place calculation
    const bool in_place      = (src1 == dst) || (src2 == dst);
    const bool src1_in_place = in_place && (src1 == dst);
    build_opts.add_option_if(in_place, "-DIN_PLACE");
    build_opts.add_option_if(src1_in_place, "-DSRC1_IN_PLACE");

    // Create kernel
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Set scale argument
    unsigned int idx = (in_place ? 2 : 3) * num_arguments_per_3D_tensor(); // Skip the src and dst parameters

    if (scale_int >= 0 && !is_quantized)
    {
        _kernel.setArg(idx++, scale_int);
    }
    else
    {
        _kernel.setArg(idx++, scale);
    }

    Window win = calculate_max_window(*dst, Steps(vec_size));
    ICLKernel::configure_internal(win);

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(dst->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(src1->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(src1->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(src1->dimension(2));
    _config_id += "_";
    _config_id += support::cpp11::to_string(src2->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(src2->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(src2->dimension(2));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(2));
}

Status ClMulKernel::validate(const ITensorInfo         *src1,
                             const ITensorInfo         *src2,
                             const ITensorInfo         *dst,
                             float                      scale,
                             ConvertPolicy              overflow_policy,
                             RoundingPolicy             rounding_policy,
                             const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src1, src2, dst);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src1, src2, dst, scale, overflow_policy, rounding_policy, act_info));

    return Status{};
}

void ClMulKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const auto src_0 =
        utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_0));
    const auto src_1 =
        utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_1));
    auto dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    ARM_COMPUTE_ERROR_ON_NULLPTR(src_0, src_1, dst);

    const TensorShape &in_shape1 = src_0->info()->tensor_shape();
    const TensorShape &in_shape2 = src_1->info()->tensor_shape();
    const TensorShape &out_shape = dst->info()->tensor_shape();

    bool can_collapse = true;
    if (std::min(in_shape1.total_size(), in_shape2.total_size()) > 1)
    {
        can_collapse = (std::min(in_shape1.num_dimensions(), in_shape2.num_dimensions()) > Window::DimZ);
        for (size_t d = Window::DimZ; can_collapse && (d < out_shape.num_dimensions()); ++d)
        {
            can_collapse = (in_shape1[d] == in_shape2[d]);
        }
    }

    bool   has_collapsed = false;
    Window collapsed =
        can_collapse ? window.collapse_if_possible(ICLKernel::window(), Window::DimZ, &has_collapsed) : window;

    const TensorShape &in_shape1_collapsed = has_collapsed ? in_shape1.collapsed_from(Window::DimZ) : in_shape1;
    const TensorShape &in_shape2_collapsed = has_collapsed ? in_shape2.collapsed_from(Window::DimZ) : in_shape2;

    Window slice        = collapsed.first_slice_window_3D();
    Window slice_input1 = slice.broadcast_if_dimension_le_one(in_shape1_collapsed);
    Window slice_input2 = slice.broadcast_if_dimension_le_one(in_shape2_collapsed);

    // Check whether it is in_place calculation
    const bool in_place = (src_0 == dst) || (src_1 == dst);
    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, src_0, slice_input1);
        add_3D_tensor_argument(idx, src_1, slice_input2);
        if (!in_place)
        {
            add_3D_tensor_argument(idx, dst, slice);
        }
        enqueue(queue, *this, slice, lws_hint());

        ARM_COMPUTE_UNUSED(collapsed.slide_window_slice_3D(slice_input1));
        ARM_COMPUTE_UNUSED(collapsed.slide_window_slice_3D(slice_input2));
    } while (collapsed.slide_window_slice_3D(slice));
}

namespace
{
constexpr unsigned int vec_size_complex = 1;

Status validate_arguments_complex(const ITensorInfo         *src1,
                                  const ITensorInfo         *src2,
                                  const ITensorInfo         *dst,
                                  const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src1, 2, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src2, 2, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src1, src2);

    const TensorShape &out_shape = TensorShape::broadcast_shape(src1->tensor_shape(), src2->tensor_shape());

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.enabled() && !is_data_type_float(dst->data_type()));

    // Validate in case of configured dst
    if (dst->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 2, DataType::F16, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src1, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(out_shape, dst->tensor_shape(), 0),
                                        "Wrong shape for dst");
    }

    return Status{};
}
} // namespace

ClComplexMulKernel::ClComplexMulKernel()
{
    _type = CLKernelType::ELEMENTWISE;
}

void ClComplexMulKernel::configure(const CLCompileContext    &compile_context,
                                   ITensorInfo               *src1,
                                   ITensorInfo               *src2,
                                   ITensorInfo               *dst,
                                   const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src1, src2, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_complex(src1, src2, dst, act_info));

    auto padding_info = get_padding_info({src1, src2, dst});

    const TensorShape &out_shape = TensorShape::broadcast_shape(src1->tensor_shape(), src2->tensor_shape());
    auto_init_if_empty(*dst, src1->clone()->set_tensor_shape(out_shape));

    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(dst->data_type()));
    if (act_info.enabled())
    {
        build_opts.add_option("-DACTIVATION_TYPE=" + lower_string(string_from_activation_func(act_info.activation())));
        build_opts.add_option("-DA_VAL=" + float_to_string_with_full_precision(act_info.a()));
        build_opts.add_option("-DB_VAL=" + float_to_string_with_full_precision(act_info.b()));
    }

    // Create kernel
    _kernel = create_kernel(compile_context, "pixelwise_mul_complex", build_opts.options());

    Window win = calculate_max_window(*dst, Steps(vec_size_complex));
    ICLKernel::configure_internal(win);

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status ClComplexMulKernel::validate(const ITensorInfo         *src1,
                                    const ITensorInfo         *src2,
                                    const ITensorInfo         *dst,
                                    const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src1, src2, dst);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_complex(src1, src2, dst, act_info));

    return Status{};
}

void ClComplexMulKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const auto src_0 =
        utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_0));
    const auto src_1 =
        utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_1));
    auto dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    const TensorShape &in_shape1 = src_0->info()->tensor_shape();
    const TensorShape &in_shape2 = src_1->info()->tensor_shape();
    const TensorShape &out_shape = dst->info()->tensor_shape();

    bool can_collapse = true;
    if (std::min(in_shape1.total_size(), in_shape2.total_size()) > 1)
    {
        can_collapse = (std::min(in_shape1.num_dimensions(), in_shape2.num_dimensions()) > Window::DimZ);
        for (size_t d = Window::DimZ; can_collapse && (d < out_shape.num_dimensions()); ++d)
        {
            can_collapse = (in_shape1[d] == in_shape2[d]);
        }
    }

    bool   has_collapsed = false;
    Window collapsed =
        can_collapse ? window.collapse_if_possible(ICLKernel::window(), Window::DimZ, &has_collapsed) : window;

    const TensorShape &in_shape1_collapsed = has_collapsed ? in_shape1.collapsed_from(Window::DimZ) : in_shape1;
    const TensorShape &in_shape2_collapsed = has_collapsed ? in_shape2.collapsed_from(Window::DimZ) : in_shape2;

    Window slice        = collapsed.first_slice_window_3D();
    Window slice_input1 = slice.broadcast_if_dimension_le_one(in_shape1_collapsed);
    Window slice_input2 = slice.broadcast_if_dimension_le_one(in_shape2_collapsed);

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, src_0, slice_input1);
        add_3D_tensor_argument(idx, src_1, slice_input2);
        add_3D_tensor_argument(idx, dst, slice);
        enqueue(queue, *this, slice, lws_hint());

        ARM_COMPUTE_UNUSED(collapsed.slide_window_slice_3D(slice_input1));
        ARM_COMPUTE_UNUSED(collapsed.slide_window_slice_3D(slice_input2));
    } while (collapsed.slide_window_slice_3D(slice));
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
