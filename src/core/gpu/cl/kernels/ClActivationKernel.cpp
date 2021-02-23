/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#include "src/core/gpu/cl/kernels/ClActivationKernel.h"

#include "arm_compute/core/CL/CLCoreRuntimeContext.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/Cast.h"

#include "support/StringSupport.h"

#include <set>

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
namespace
{
Status validate_arguments(const ITensorInfo *src, const ITensorInfo *dst, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::QSYMM16, DataType::F16, DataType::F32);

    static std::set<ActivationLayerInfo::ActivationFunction> quantized_supported_activations =
    {
        ActivationLayerInfo::ActivationFunction::RELU,
        ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
        ActivationLayerInfo::ActivationFunction::BOUNDED_RELU,
        ActivationLayerInfo::ActivationFunction::LOGISTIC,
        ActivationLayerInfo::ActivationFunction::TANH,
        ActivationLayerInfo::ActivationFunction::HARD_SWISH,
        ActivationLayerInfo::ActivationFunction::LEAKY_RELU,
    };
    const DataType                                data_type = src->data_type();
    const QuantizationInfo                       &oq_info   = (dst != nullptr) ? dst->quantization_info() : src->quantization_info();
    const ActivationLayerInfo::ActivationFunction f_act     = act_info.activation();

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(is_data_type_quantized(data_type) && (quantized_supported_activations.count(f_act) == 0),
                                    "For Quantized data type only hard swish, leaky relu, tanh, logistic, relu and lower/upper bounded relu are supported");

    ARM_COMPUTE_RETURN_ERROR_ON(data_type == DataType::QASYMM8 && (f_act == ActivationLayerInfo::ActivationFunction::TANH) && (oq_info != QuantizationInfo(1.f / 128.f, 128)));
    ARM_COMPUTE_RETURN_ERROR_ON(data_type == DataType::QASYMM8 && (f_act == ActivationLayerInfo::ActivationFunction::LOGISTIC) && (oq_info != QuantizationInfo(1.f / 256.f, 0)));

    ARM_COMPUTE_RETURN_ERROR_ON(is_data_type_quantized_symmetric(data_type) && (f_act == ActivationLayerInfo::ActivationFunction::TANH) && (oq_info != QuantizationInfo(1.f / 32768.f, 0)));
    ARM_COMPUTE_RETURN_ERROR_ON(is_data_type_quantized_symmetric(data_type) && (f_act == ActivationLayerInfo::ActivationFunction::LOGISTIC) && (oq_info != QuantizationInfo(1.f / 32768.f, 0)));

    ARM_COMPUTE_RETURN_ERROR_ON(data_type == DataType::QASYMM8_SIGNED && (f_act == ActivationLayerInfo::ActivationFunction::TANH) && (oq_info != QuantizationInfo(1.f / 128.f, 0)));
    ARM_COMPUTE_RETURN_ERROR_ON(data_type == DataType::QASYMM8_SIGNED && (f_act == ActivationLayerInfo::ActivationFunction::LOGISTIC) && (oq_info != QuantizationInfo(1.f / 256.f, -128)));

    // Checks performed when destination is configured
    if((dst != nullptr) && (dst->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
    }

    return Status{};
}
} // namespace

ClActivationKernel::ClActivationKernel()
    : _run_in_place(false)
{
}

void ClActivationKernel::configure(const ClCompileContext &compile_context, ITensorInfo *src, ITensorInfo *dst, ActivationLayerInfo act_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src);

    auto padding_info = get_padding_info({ src, dst });

    _run_in_place = (dst == nullptr) || (dst == src);

    if(dst != nullptr)
    {
        // Destination auto inizialitation if not yet initialized
        auto_init_if_empty(*dst, *src->clone());
    }

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, (dst != nullptr) ? dst : nullptr, act_info));

    const unsigned int num_elems_processed_per_iteration = adjust_vec_size(16 / src->element_size(), src->dimension(0));

    const DataType dt      = src->data_type();
    float          a_const = act_info.a();
    float          b_const = act_info.b();

    const ActivationLayerInfo::ActivationFunction f_act        = act_info.activation();
    const bool                                    is_quantized = is_data_type_quantized(dt);
    const bool                                    perform_activation_in_float =
        (f_act == ActivationLayerInfo::ActivationFunction::LOGISTIC)
        || (f_act == ActivationLayerInfo::ActivationFunction::TANH)
        || (f_act == ActivationLayerInfo::ActivationFunction::HARD_SWISH)
        || (f_act == ActivationLayerInfo::ActivationFunction::LEAKY_RELU);

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option_if(perform_activation_in_float, "-DFLOAT_DOMAIN");
    build_opts.add_option_if(_run_in_place, "-DIN_PLACE");
    build_opts.add_option("-DACT=" + lower_string(string_from_activation_func(f_act)));
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(dt));
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration));
    build_opts.add_option("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(src->dimension(0) % num_elems_processed_per_iteration));

    std::string kernel_name = std::string("activation_layer");

    // Set quantization info build options
    if(is_quantized)
    {
        const UniformQuantizationInfo iq_info = src->quantization_info().uniform();

        if(!perform_activation_in_float)
        {
            int a_const_int = 0;
            int b_const_int = 0;

            // Create quantized version of constants a, b if needed
            switch(dt)
            {
                case DataType::QASYMM8:
                {
                    a_const_int = quantize_qasymm8(a_const, iq_info);
                    b_const_int = quantize_qasymm8(b_const, iq_info);
                }
                break;
                case DataType::QASYMM8_SIGNED:
                {
                    a_const_int = quantize_qasymm8_signed(a_const, iq_info);
                    b_const_int = quantize_qasymm8_signed(b_const, iq_info);
                }
                break;
                case DataType::QSYMM16:
                {
                    a_const_int = quantize_qsymm16(a_const, iq_info);
                    b_const_int = quantize_qsymm16(b_const, iq_info);
                }
                break;
                default:
                    break;
            }
            build_opts.add_option(("-DA_VAL=" + support::cpp11::to_string(a_const_int)));
            build_opts.add_option(("-DB_VAL=" + support::cpp11::to_string(b_const_int)));
        }
        else
        {
            build_opts.add_option(("-DA_VAL=" + float_to_string_with_full_precision(a_const)));
            build_opts.add_option(("-DB_VAL=" + float_to_string_with_full_precision(b_const)));
        }

        // Quantized value of 0 corresponds to the offset o1
        build_opts.add_option(("-DCONST_0=" + (is_data_type_quantized_asymmetric(dt) ? support::cpp11::to_string(iq_info.offset) : "0")));
        build_opts.add_option(("-DS1_VAL=" + float_to_string_with_full_precision(iq_info.scale)));
        build_opts.add_option_if(is_data_type_quantized_asymmetric(dt), "-DO1_VAL=" + support::cpp11::to_string(iq_info.offset));

        // Set correct kernel name
        kernel_name += perform_activation_in_float ? std::string("_quant_f32") : std::string("_quant");

        // Set scale and offset of the source and destination if they have different quantization info
        if(dst != nullptr)
        {
            const UniformQuantizationInfo oq_info = dst->quantization_info().uniform();

            if(iq_info != oq_info)
            {
                build_opts.add_option(("-DS2_VAL=" + float_to_string_with_full_precision(oq_info.scale)));
                build_opts.add_option_if(is_data_type_quantized_asymmetric(dt), "-DO2_VAL=" + support::cpp11::to_string(oq_info.offset));
            }
        }
    }
    else
    {
        // Set A, B constants in build options for float types
        build_opts.add_option(("-DA_VAL=" + float_to_string_with_full_precision(a_const)));
        build_opts.add_option(("-DB_VAL=" + float_to_string_with_full_precision(b_const)));
    }

    // Create kernel
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Configure kernel window
    Window win = calculate_max_window(*src, Steps(num_elems_processed_per_iteration));
    ICLKernel::configure_internal(win);

    // Set config_id for enabling LWS tuning
    _config_id = "activation_layer_";
    _config_id += lower_string(string_from_data_type(dt));
    _config_id += "_";
    _config_id += support::cpp11::to_string(src->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(src->dimension(1));

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status ClActivationKernel::validate(const ITensorInfo *src, const ITensorInfo *dst, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst, act_info));
    return Status{};
}

void ClActivationKernel::run_op(ITensorPack &tensors, const Window &window, ::cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const auto src = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    auto       dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));
    ARM_COMPUTE_ERROR_ON(_run_in_place && src != dst);

    Window collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice     = collapsed.first_slice_window_3D();

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, src, slice);
        if(!_run_in_place)
        {
            add_3D_tensor_argument(idx, dst, slice);
        }
        enqueue(queue, *this, slice, lws_hint());
    }
    while(collapsed.slide_window_slice_3D(slice));
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
