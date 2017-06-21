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
#include "arm_compute/core/NEON/kernels/NEWeightsReshapeKernel.h"

#include "arm_compute/core/Dimensions.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

using namespace arm_compute;

namespace
{
template <typename T>
void weights_reshape(const ITensor *input, const ITensor *bias, ITensor *output, const Window &window)
{
    const unsigned int kernel_size_x   = input->info()->dimension(0);
    const unsigned int kernel_size_y   = input->info()->dimension(1);
    const unsigned int kernel_depth    = input->info()->dimension(2);
    const unsigned int input_stride_x  = input->info()->strides_in_bytes().x();
    const unsigned int input_stride_y  = input->info()->strides_in_bytes().y();
    const unsigned int input_stride_z  = input->info()->strides_in_bytes().z();
    const unsigned int output_stride_y = output->info()->strides_in_bytes().y();

    // Create iterators
    Iterator in(input, window);
    execute_window_loop(window, [&](const Coordinates & id)
    {
        // Get column index
        const int kernel_idx = id[3];
        const int kernel_idz = id[4];

        // Setup pointers
        const uint8_t *tmp_input_ptr        = in.ptr();
        uint8_t       *tmp_output_ptr       = output->ptr_to_element(Coordinates(kernel_idx, 0, kernel_idz));
        const uint8_t *curr_input_row_ptr   = tmp_input_ptr;
        const uint8_t *curr_input_depth_ptr = tmp_input_ptr;

        // Linearize volume
        for(unsigned int d = 0; d < kernel_depth; ++d)
        {
            for(unsigned int j = 0; j < kernel_size_y; ++j)
            {
                for(unsigned int i = 0; i < kernel_size_x; ++i)
                {
                    *(reinterpret_cast<T *>(tmp_output_ptr)) = *(reinterpret_cast<const T *>(tmp_input_ptr));
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
        if(bias != nullptr)
        {
            *(reinterpret_cast<T *>(tmp_output_ptr)) = *(reinterpret_cast<const T *>(bias->ptr_to_element(Coordinates(kernel_idx, kernel_idz))));
        }
    },
    in);
}
} // namespace

NEWeightsReshapeKernel::NEWeightsReshapeKernel()
    : _func(nullptr), _input(nullptr), _bias(nullptr), _output(nullptr)
{
}

void NEWeightsReshapeKernel::configure(const ITensor *input, const ITensor *bias, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32, DataType::QS8);
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(0) != input->info()->dimension(1));

    const DataType dt                   = input->info()->data_type();
    const int      fixed_point_position = input->info()->fixed_point_position();

    TensorShape output_shape{ input->info()->tensor_shape() };
    output_shape.collapse(3);
    const size_t tmp_dim = output_shape[0];
    output_shape.set(0, output_shape[1]);
    output_shape.set(1, tmp_dim + (bias != nullptr ? 1 : 0));

    // Set data type and shape for output tensor if not yet configured
    set_data_type_if_unknown(*output->info(), dt);
    set_fixed_point_position_if_zero(*output->info(), fixed_point_position);

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(output->info()->tensor_shape(), output_shape);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F32, DataType::QS8);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);

    if(bias != nullptr)
    {
        TensorShape bias_shape{ input->info()->tensor_shape()[3] };

        // Set data type and shape for bias tensor if not yet configured
        set_data_type_if_unknown(*bias->info(), dt);
        set_fixed_point_position_if_zero(*bias->info(), fixed_point_position);
        set_shape_if_empty(*bias->info(), bias_shape);

        ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(bias->info()->tensor_shape(), bias_shape);
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(bias, 1, DataType::F32, DataType::QS8);
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, bias);
        ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);
    }

    _input  = input;
    _bias   = bias;
    _output = output;

    switch(_input->info()->data_type())
    {
        case DataType::F32:
        {
            _func = &weights_reshape<uint32_t>;
            break;
        }
        case DataType::QS8:
        {
            _func = &weights_reshape<uint8_t>;
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR_ON("Data type not supported");
            break;
        }
    }

    // Configure kernel
    Window window = calculate_max_window(*input->info(), Steps());
    window.set(Window::DimX, Window::Dimension(0, _input->info()->dimension(0), _input->info()->dimension(0)));
    window.set(Window::DimY, Window::Dimension(0, _input->info()->dimension(1), _input->info()->dimension(1)));
    window.set(Window::DimZ, Window::Dimension(0, _input->info()->dimension(2), _input->info()->dimension(2)));

    // The NEConvolutionLayerWeightsReshapeKernel doesn't need padding so update_window_and_padding() can be skipped
    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));

    INEKernel::configure(window);
}

void NEWeightsReshapeKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    (*_func)(_input, _bias, _output, window);
}
