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
#include "src/core/NEON/kernels/NEWeightsReshapeKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
namespace
{
TensorShape get_output_shape(const ITensorInfo *input, bool has_bias)
{
    TensorShape output_shape{ input->tensor_shape() };

    output_shape.collapse(3);
    const size_t tmp_dim = output_shape[0];
    output_shape.set(0, output_shape[1]);
    output_shape.set(1, tmp_dim + (has_bias ? 1 : 0));

    return output_shape;
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *biases, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    //Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input) is not needed here as this kernel doesn't use Neon FP16 instructions.
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);

    if(biases != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(is_data_type_quantized_asymmetric(input->data_type()));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
        ARM_COMPUTE_RETURN_ERROR_ON((input->num_dimensions() == 4) && (biases->num_dimensions() != 1));
        ARM_COMPUTE_RETURN_ERROR_ON((input->num_dimensions() == 5) && (biases->num_dimensions() != 2));
        ARM_COMPUTE_RETURN_ERROR_ON((input->num_dimensions() == 4) && (biases->dimension(0) != input->tensor_shape()[3]));
        ARM_COMPUTE_RETURN_ERROR_ON((input->num_dimensions() == 5) && (biases->dimension(0) != input->tensor_shape()[3] || biases->dimension(1) != input->tensor_shape()[4]));
    }

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), get_output_shape(input, biases != nullptr));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    Window window = calculate_max_window(*input, Steps());
    window.set(Window::DimX, Window::Dimension(0, input->dimension(0), input->dimension(0)));
    window.set(Window::DimY, Window::Dimension(0, input->dimension(1), input->dimension(1)));
    window.set(Window::DimZ, Window::Dimension(0, input->dimension(2), input->dimension(2)));

    // The NEConvolutionLayerWeightsReshapeKernel doesn't need padding so update_window_and_padding() can be skipped
    output->set_valid_region(ValidRegion(Coordinates(), output->tensor_shape()));

    return std::make_pair(Status{}, window);
}
} // namespace

NEWeightsReshapeKernel::NEWeightsReshapeKernel()
    : _input(nullptr), _bias(nullptr), _output(nullptr)
{
}

void NEWeightsReshapeKernel::configure(const ITensor *input, const ITensor *bias, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Output tensor auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(get_output_shape(input->info(), (bias != nullptr))));

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(),
                                                  (bias != nullptr) ? bias->info() : nullptr,
                                                  output->info()));

    _input  = input;
    _bias   = bias;
    _output = output;

    // Configure kernel
    auto win_config = validate_and_configure_window(input->info(), output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NEWeightsReshapeKernel::validate(const ITensorInfo *input, const ITensorInfo *biases, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, biases, output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get()).first);

    return Status{};
}

void NEWeightsReshapeKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const unsigned int kernel_size_x   = _input->info()->dimension(0);
    const unsigned int kernel_size_y   = _input->info()->dimension(1);
    const unsigned int kernel_depth    = _input->info()->dimension(2);
    const unsigned int input_stride_x  = _input->info()->strides_in_bytes().x();
    const unsigned int input_stride_y  = _input->info()->strides_in_bytes().y();
    const unsigned int input_stride_z  = _input->info()->strides_in_bytes().z();
    const unsigned int output_stride_y = _output->info()->strides_in_bytes().y();

    // Create iterators
    Iterator in(_input, window);
    execute_window_loop(window, [&](const Coordinates & id)
    {
        // Get column index
        const int kernel_idx = id[3];
        const int kernel_idz = id[4];

        // Setup pointers
        const uint8_t *tmp_input_ptr        = in.ptr();
        uint8_t       *tmp_output_ptr       = _output->ptr_to_element(Coordinates(kernel_idx, 0, kernel_idz));
        const uint8_t *curr_input_row_ptr   = tmp_input_ptr;
        const uint8_t *curr_input_depth_ptr = tmp_input_ptr;

        // Linearize volume
        for(unsigned int d = 0; d < kernel_depth; ++d)
        {
            for(unsigned int j = 0; j < kernel_size_y; ++j)
            {
                for(unsigned int i = 0; i < kernel_size_x; ++i)
                {
                    std::memcpy(tmp_output_ptr, tmp_input_ptr, _input->info()->element_size());
                    tmp_input_ptr += input_stride_x;
                    tmp_output_ptr += output_stride_y;
                }
                curr_input_row_ptr += input_stride_y;
                tmp_input_ptr = curr_input_row_ptr;
            }
            curr_input_depth_ptr += input_stride_z;
            curr_input_row_ptr = curr_input_depth_ptr;
            tmp_input_ptr      = curr_input_depth_ptr;
        }

        // Add bias
        if(_bias != nullptr)
        {
            std::memcpy(tmp_output_ptr, _bias->ptr_to_element(Coordinates(kernel_idx, kernel_idz)), _input->info()->element_size());
        }
    },
    in);
}
} // namespace arm_compute
