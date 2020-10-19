/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/core/CL/kernels/CLQuantizationLayerKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F32, DataType::F16);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);

    // Output must always be initialized
    ARM_COMPUTE_RETURN_ERROR_ON(output->tensor_shape().total_size() == 0);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::QASYMM16);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    // Configure kernel window
    Window win = calculate_max_window(*input, Steps());

    const int  vec_size_x     = 16 / input->element_size();
    const int  input_width_x  = input->tensor_shape().x();
    const bool multi_access_x = (input_width_x / vec_size_x > 0);
    if(multi_access_x)
    {
        win.set(Window::DimX, Window::Dimension(win.x().start(), ceil_to_multiple(win.x().end(), vec_size_x), vec_size_x));
    }

    Coordinates coord;
    coord.set_num_dimensions(output->num_dimensions());
    output->set_valid_region(ValidRegion(coord, output->tensor_shape()));

    return std::make_pair(Status{}, win);
}
} // namespace

CLQuantizationLayerKernel::CLQuantizationLayerKernel()
    : _input(nullptr), _output(nullptr)
{
}

void CLQuantizationLayerKernel::configure(const ICLTensor *input, ICLTensor *output)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output);
}

void CLQuantizationLayerKernel::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info()));

    _input  = input;
    _output = output;

    const int  vec_size_x     = 16 / input->info()->element_size();
    const int  input_width_x  = input->info()->tensor_shape().x();
    const bool multi_access_x = (input_width_x / vec_size_x > 0);

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

    const UniformQuantizationInfo qinfo            = output->info()->quantization_info().uniform();
    const DataType                output_data_type = output->info()->data_type();

    float   scale_to_apply  = qinfo.scale;
    int32_t offset_to_apply = qinfo.offset;
    if(is_data_type_quantized_asymmetric(_input->info()->data_type()))
    {
        /*
         * In case of requantization of a quantized input tensor to an output tensor with another quantization
         * instead of of apply dequantization and then a quantization functions, we just compute new scale and
         * offset to apply.
         *
         * Assuming:
         *   - q_i as input quantized value
         *   - q_o as output quantized value
         *   - z_i as input quantization offset value
         *   - z_o as output quantization offset value
         *   - s_i as input quantization scale value
         *   - s_o as output quantization scale value
         *   - z_n as new quantization offset value
         *   - s_n as new quantization scale value
         *
         * q_o = ( q_i - z_i ) * s_i / s_o + z_o
         *
         * We can rewrite the formula as:
         *
         * q_o = ( q_i * s_i / s_o ) - z_i * s_i / s_o + z_o
         *
         * q_o = q_i / s_n + z_n
         *
         * Where:
         *
         * s_n = s_o / s_i
         *
         * z_n = - z_i * s_i / s_o + z_o
         *
         */
        const UniformQuantizationInfo qinfo_in = _input->info()->quantization_info().uniform();
        scale_to_apply /= qinfo_in.scale;
        // In order to minimize flooring we convert the offset to a float,
        // then compute the new offset in the float domain,
        // finally we convert it back as int32_t
        offset_to_apply -= static_cast<int32_t>(static_cast<float>(qinfo_in.offset) * qinfo_in.scale / qinfo.scale);
    }

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option_if(is_data_type_float(_input->info()->data_type()), "-DIS_FLOAT");
    build_opts.add_option("-DSCALE=" + float_to_string_with_full_precision(scale_to_apply));
    build_opts.add_option("-DOFFSET=" + support::cpp11::to_string(offset_to_apply));
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(vec_size_x));
    build_opts.add_option("-DDATA_TYPE_IN=" + get_cl_type_from_data_type(input->info()->data_type()));
    build_opts.add_option("-DDATA_TYPE_OUT=" + get_cl_type_from_data_type(output_data_type));
    build_opts.add_option_if(multi_access_x, "-DLAST_ACCESSED_X=" + support::cpp11::to_string(std::max<int>(input_width_x - vec_size_x, 0)));
    std::pair<int, int> min_max_quant_values = quantization::get_min_max_values_from_quantized_data_type(output_data_type);
    build_opts.add_option("-DMIN_QUANT_VAL=" + support::cpp11::to_string(min_max_quant_values.first));
    build_opts.add_option("-DMAX_QUANT_VAL=" + support::cpp11::to_string(min_max_quant_values.second));

    _kernel = create_kernel(compile_context, "quantization_layer", build_opts.options());
}

Status CLQuantizationLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get()).first);

    return Status{};
}

void CLQuantizationLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), 3);
    Window slice            = window_collapsed.first_slice_window_3D();

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(window_collapsed.slide_window_slice_3D(slice));
}
} // namespace arm_compute
