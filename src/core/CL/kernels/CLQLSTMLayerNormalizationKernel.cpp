/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#include "src/core/CL/kernels/CLQLSTMLayerNormalizationKernel.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace
{
QuantizationInfo compute_output_qinfo()
{
    return QuantizationInfo(1.f / 4096);
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input);
    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output, *input);
    output->set_quantization_info(compute_output_qinfo());

    const uint32_t temp_num_elems_processed_per_iteration = max_cl_vector_width / input->element_size();
    /* If width is less then step, then make step same as width to avoid global size being step instead of actual width. */
    /* Or we should fix in arm_compute::enqueue() or arm_compute::calculate_max_window(). */
    const uint32_t num_elems_processed_per_iteration = (input->dimension(0) < temp_num_elems_processed_per_iteration) ? input->dimension(0) : temp_num_elems_processed_per_iteration;

    // This kernel doesn't need padding
    Window win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));

    return std::make_pair(Status{}, win);
}
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *weight, const ITensorInfo *bias)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weight, bias, output);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->num_dimensions() > 2, "Input tensor cannot have more than 2 dimensions");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weight->num_dimensions() > 1, "Weight tensor cannot have more than 1 dimensions");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(bias->num_dimensions() > 1, "Bias tensor cannot have more than 1 dimensions");

    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QSYMM16);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weight, 1, DataType::QSYMM16);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(bias, 1, DataType::S32);

    ARM_COMPUTE_RETURN_ERROR_ON(input->tensor_shape().x() != weight->tensor_shape().x());
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(weight, bias);

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }
    return Status{};
}
} // namespace

CLQLSTMLayerNormalizationKernel::CLQLSTMLayerNormalizationKernel()
    : _input(nullptr), _weight(nullptr), _bias(nullptr), _output(nullptr)
{
}

void CLQLSTMLayerNormalizationKernel::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, const ICLTensor *weight, const ICLTensor *bias)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weight, bias, output);
    auto padding_info = get_padding_info({ input, weight, bias, output });

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), weight->info(), bias->info()));

    _input  = input;
    _weight = weight;
    _bias   = bias;
    _output = output;

    const uint32_t num_elems_processed_per_iteration = max_cl_vector_width / input->info()->element_size();

    int32_t                       output_multiplier{};
    int32_t                       output_shift{};
    const UniformQuantizationInfo quan_info = _weight->info()->quantization_info().uniform();
    const Status                  status    = quantization::calculate_quantized_multiplier(quan_info.scale, &output_multiplier, &output_shift);
    output_shift *= -1;

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration));
    build_opts.add_option("-DWIDTH=" + support::cpp11::to_string(input->info()->dimension(0)));
    build_opts.add_option("-DOUTPUT_MULTIPLIER=" + support::cpp11::to_string(output_multiplier));
    build_opts.add_option("-DOUTPUT_SHIFT=" + support::cpp11::to_string(output_shift));
    build_opts.add_option("-DMIN_BOUND=" + support::cpp11::to_string(std::get<0>(quantization::get_min_max_values_from_quantized_data_type(input->info()->data_type()))));
    build_opts.add_option("-DMAX_BOUND=" + support::cpp11::to_string(std::get<1>(quantization::get_min_max_values_from_quantized_data_type(input->info()->data_type()))));

    // Create kernel
    _kernel = create_kernel(compile_context, "qlstm_layer_normalization", build_opts.options());

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

    // Set config_id for enabling LWS tuning
    _config_id = "qlstm_layer_normalization_";
    _config_id += lower_string(string_from_data_type(input->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(1));
    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

void CLQLSTMLayerNormalizationKernel::configure(const ICLTensor *input, ICLTensor *output, const ICLTensor *weight, const ICLTensor *bias)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, weight, bias);
}

Status CLQLSTMLayerNormalizationKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *weight, const ITensorInfo *bias)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, weight, bias));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get()).first);
    return Status{};
}

void CLQLSTMLayerNormalizationKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice = window.first_slice_window_2D();
    // Set slice step equal to width to force gws[0] to 1, as each thread normalizes across all rows
    slice.set_dimension_step(Window::DimX, _input->info()->dimension(0));

    Window weight_window;
    Window weight_slice;

    weight_window.use_tensor_dimensions(_weight->info()->tensor_shape());
    weight_slice = weight_window.first_slice_window_1D();

    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input, slice);
        add_1D_tensor_argument(idx, _weight, weight_slice);
        add_1D_tensor_argument(idx, _bias, weight_slice);
        add_2D_tensor_argument(idx, _output, slice);

        enqueue(queue, *this, slice, lws_hint());
    }
    while(window.slide_window_slice_2D(slice));
}
} // namespace arm_compute
