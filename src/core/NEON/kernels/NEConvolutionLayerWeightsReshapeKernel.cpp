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
#include "arm_compute/core/NEON/kernels/NEConvolutionLayerWeightsReshapeKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

using namespace arm_compute;

NEConvolutionLayerWeightsReshapeKernel::NEConvolutionLayerWeightsReshapeKernel()
    : _input(nullptr), _bias(nullptr), _output(nullptr), _has_bias(false)
{
}

void NEConvolutionLayerWeightsReshapeKernel::configure(const ITensor *input, const ITensor *bias, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F32);
    if(bias != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(bias, 1, DataType::F32);
    }
    ARM_COMPUTE_ERROR_ON(input->info()->num_dimensions() > 4);
    ARM_COMPUTE_ERROR_ON(output->info()->num_dimensions() > 2);
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(0) != input->info()->dimension(1));

    _input    = input;
    _bias     = bias;
    _output   = output;
    _has_bias = (bias != nullptr);

    // Configure kernel
    Window window = calculate_max_window(*input->info(), Steps());
    window.set(Window::DimX, Window::Dimension(0, _input->info()->dimension(0), _input->info()->dimension(0)));
    window.set(Window::DimY, Window::Dimension(0, _input->info()->dimension(1), _input->info()->dimension(1)));
    window.set(Window::DimZ, Window::Dimension(0, _input->info()->dimension(2), _input->info()->dimension(2)));

    // The NEConvolutionLayerWeightsReshapeKernel doesn't need padding so update_window_and_padding() can be skipped
    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));

    INEKernel::configure(window);
}

void NEConvolutionLayerWeightsReshapeKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const unsigned int kernel_size     = _input->info()->dimension(0);
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

        // Setup pointers
        auto tmp_input_ptr        = in.ptr();
        auto tmp_output_ptr       = _output->ptr_to_element(Coordinates(kernel_idx, 0));
        auto curr_input_row_ptr   = tmp_input_ptr;
        auto curr_input_depth_ptr = tmp_input_ptr;

        // Linearize volume
        for(unsigned int d = 0; d < kernel_depth; ++d)
        {
            for(unsigned int j = 0; j < kernel_size; ++j)
            {
                for(unsigned int i = 0; i < kernel_size; ++i)
                {
                    *(reinterpret_cast<float *>(tmp_output_ptr)) = *(reinterpret_cast<float *>(tmp_input_ptr));
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
        if(_has_bias)
        {
            *(reinterpret_cast<float *>(tmp_output_ptr)) = *(reinterpret_cast<float *>(_bias->ptr_to_element(Coordinates(kernel_idx, 0))));
        }
    },
    in);
}
