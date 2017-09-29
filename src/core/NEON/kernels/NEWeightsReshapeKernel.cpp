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
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);

    const int          fixed_point_position = input->info()->fixed_point_position();
    const DataType     dt                   = input->info()->data_type();
    const TensorShape &input_shape          = input->info()->tensor_shape();
    TensorShape        output_shape{ input_shape };
    output_shape.collapse(3);

    const size_t tmp_dim = output_shape[0];
    output_shape.set(0, output_shape[1]);
    output_shape.set(1, tmp_dim + (bias != nullptr ? 1 : 0));

    // Output tensor auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, dt, fixed_point_position);

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(output->info()->tensor_shape(), output_shape);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);

    if(bias != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, bias);
        ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, bias);
        ARM_COMPUTE_ERROR_ON((input->info()->num_dimensions() == 4) && (bias->info()->num_dimensions() != 1));
        ARM_COMPUTE_ERROR_ON((input->info()->num_dimensions() == 5) && (bias->info()->num_dimensions() != 2));
        ARM_COMPUTE_ERROR_ON((input->info()->num_dimensions() == 4) && (bias->info()->dimension(0) != input->info()->tensor_shape()[3]));
        ARM_COMPUTE_ERROR_ON((input->info()->num_dimensions() == 5) && (bias->info()->dimension(0) != input->info()->tensor_shape()[3] || bias->info()->dimension(1) != input->info()->tensor_shape()[4]));
    }

    _input  = input;
    _bias   = bias;
    _output = output;

    switch(_input->info()->element_size())
    {
        case 4:
        {
            _func = &weights_reshape<uint32_t>;
            break;
        }
        case 2:
        {
            _func = &weights_reshape<uint16_t>;
            break;
        }
        case 1:
        {
            _func = &weights_reshape<uint8_t>;
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR_ON("Element size not supported");
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

void NEWeightsReshapeKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    (*_func)(_input, _bias, _output, window);
}
