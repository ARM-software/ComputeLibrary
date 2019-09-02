/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLActivationLayerKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/helpers/float_ops.h"
#include "support/ToolchainSupport.h"

#include <cmath>
#include <set>

using namespace arm_compute;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::QASYMM8, DataType::QSYMM16, DataType::F16, DataType::F32);

    static std::set<ActivationLayerInfo::ActivationFunction> quantized_supported_activations =
    {
        ActivationLayerInfo::ActivationFunction::RELU,
        ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
        ActivationLayerInfo::ActivationFunction::BOUNDED_RELU,
        ActivationLayerInfo::ActivationFunction::LOGISTIC,
        ActivationLayerInfo::ActivationFunction::TANH
    };
    const DataType                                data_type = input->data_type();
    const QuantizationInfo                       &oq_info   = (output != nullptr) ? output->quantization_info() : input->quantization_info();
    const ActivationLayerInfo::ActivationFunction f_act     = act_info.activation();

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(is_data_type_quantized(data_type) && (quantized_supported_activations.count(f_act) == 0),
                                    "For Quantized data type only tanh, logistic, relu and lower/upper bounded relu are supported");

    ARM_COMPUTE_RETURN_ERROR_ON(is_data_type_quantized_asymmetric(data_type) && (f_act == ActivationLayerInfo::ActivationFunction::TANH) && (oq_info != QuantizationInfo(1.f / 128.f, 128)));
    ARM_COMPUTE_RETURN_ERROR_ON(is_data_type_quantized_asymmetric(data_type) && (f_act == ActivationLayerInfo::ActivationFunction::LOGISTIC) && (oq_info != QuantizationInfo(1.f / 256.f, 0)));

    ARM_COMPUTE_RETURN_ERROR_ON(is_data_type_quantized_symmetric(data_type) && (f_act == ActivationLayerInfo::ActivationFunction::TANH) && (oq_info != QuantizationInfo(1.f / 32768.f, 0)));
    ARM_COMPUTE_RETURN_ERROR_ON(is_data_type_quantized_symmetric(data_type) && (f_act == ActivationLayerInfo::ActivationFunction::LOGISTIC) && (oq_info != QuantizationInfo(1.f / 32768.f, 0)));

    // Checks performed when output is configured
    if((output != nullptr) && (output->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    if(output != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
        // Output auto inizialitation if not yet initialized
        auto_init_if_empty(*output, *input);
    }

    const unsigned int num_elems_processed_per_iteration = 16 / input->element_size();

    Window win            = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
    bool   window_changed = false;

    if(output != nullptr)
    {
        AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
        AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
        window_changed = update_window_and_padding(win, input_access, output_access);
        output_access.set_valid_region(win, input->valid_region());
    }
    else
    {
        window_changed = update_window_and_padding(win,
                                                   AccessWindowHorizontal(input, 0, num_elems_processed_per_iteration));
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

CLActivationLayerKernel::CLActivationLayerKernel()
    : _input(nullptr), _output(nullptr), _run_in_place(false)
{
}

void CLActivationLayerKernel::configure(ICLTensor *input, ICLTensor *output, ActivationLayerInfo act_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input);

    _run_in_place = (output == nullptr) || (output == input);

    if(output != nullptr)
    {
        // Output auto inizialitation if not yet initialized
        auto_init_if_empty(*output->info(),
                           *input->info()->clone());
    }

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), (output != nullptr) ? output->info() : nullptr, act_info));

    const unsigned int num_elems_processed_per_iteration = 16 / input->info()->element_size();
    const DataType     dt                                = input->info()->data_type();
    float              a_const                           = act_info.a();
    float              b_const                           = act_info.b();
    int                a_const_int                       = 0;
    int                b_const_int                       = 0;

    const ActivationLayerInfo::ActivationFunction f_act                       = act_info.activation();
    const bool                                    is_quantized                = is_data_type_quantized(dt);
    const bool                                    perform_activation_in_float = (f_act == ActivationLayerInfo::ActivationFunction::LOGISTIC) || (f_act == ActivationLayerInfo::ActivationFunction::TANH);

    // Create quantized version of constants a, b if needed
    if(dt == DataType::QASYMM8)
    {
        const UniformQuantizationInfo iq_info = input->info()->quantization_info().uniform();
        a_const_int                           = quantize_qasymm8(a_const, iq_info);
        b_const_int                           = quantize_qasymm8(b_const, iq_info);
    }
    else if(dt == DataType::QSYMM16)
    {
        const UniformQuantizationInfo iq_info = input->info()->quantization_info().uniform();
        a_const_int                           = quantize_qsymm16(a_const, iq_info);
        b_const_int                           = quantize_qsymm16(b_const, iq_info);
    }

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option_if(perform_activation_in_float, "-DFLOAT_DOMAIN");
    build_opts.add_option_if(_run_in_place, "-DIN_PLACE");
    build_opts.add_option(("-DACT=" + lower_string(string_from_activation_func(f_act))));
    build_opts.add_option(("-DDATA_TYPE=" + get_cl_type_from_data_type(dt)));
    build_opts.add_option(("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration)));

    // Set A, B constants in build options
    if(is_quantized && !perform_activation_in_float)
    {
        build_opts.add_option(("-DA_VAL=" + support::cpp11::to_string(a_const_int)));
        build_opts.add_option(("-DB_VAL=" + support::cpp11::to_string(b_const_int)));
    }
    else
    {
        build_opts.add_option(("-DA_VAL=" + float_to_string_with_full_precision(a_const)));
        build_opts.add_option(("-DB_VAL=" + float_to_string_with_full_precision(b_const)));
    }

    // Set quantization info build options
    if(is_quantized)
    {
        const UniformQuantizationInfo iq_info = input->info()->quantization_info().uniform();

        // Quantized value of 0 corresponds to the offset o1
        build_opts.add_option(("-DCONST_0=" + (is_data_type_quantized_asymmetric(dt) ? support::cpp11::to_string(iq_info.offset) : "0")));
        build_opts.add_option(("-DS1_VAL=" + float_to_string_with_full_precision(iq_info.scale)));
        build_opts.add_option_if(is_data_type_quantized_asymmetric(dt), "-DO1_VAL=" + support::cpp11::to_string(iq_info.offset));

        // Set scale and offset of the input and output if they have different quantization info
        if(output != nullptr)
        {
            const UniformQuantizationInfo oq_info = output->info()->quantization_info().uniform();

            if(iq_info != oq_info)
            {
                build_opts.add_option(("-DS2_VAL=" + float_to_string_with_full_precision(oq_info.scale)));
                build_opts.add_option_if(is_data_type_quantized_asymmetric(dt), "-DO2_VAL=" + support::cpp11::to_string(oq_info.offset));
            }
        }
    }

    // Create kernel
    std::string kernel_name = std::string("activation_layer");
    if(is_quantized)
    {
        kernel_name += perform_activation_in_float ? std::string("_quant_f32") : std::string("_quant");
    }
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));

    // Make sure _kernel is initialized before calling the parent's configure
    _input  = input;
    _output = output;

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), (_run_in_place) ? nullptr : output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

    // Set config_id for enabling LWS tuning
    _config_id = "activation_layer_";
    _config_id += lower_string(string_from_data_type(dt));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(1));
}

Status CLActivationLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    const bool run_in_place = (output == nullptr) || (output == input);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, act_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), (run_in_place) ? nullptr : output->clone().get()).first);

    return Status{};
}

void CLActivationLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice     = collapsed.first_slice_window_3D();

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        if(!_run_in_place)
        {
            add_3D_tensor_argument(idx, _output, slice);
        }
        enqueue(queue, *this, slice, lws_hint());
    }
    while(collapsed.slide_window_slice_3D(slice));
}
