/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLConvolution.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/kernels/CLConvolutionKernel.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/ITensorAllocator.h"

#include <utility>

using namespace arm_compute;

void CLConvolution3x3::configure(ICLTensor *input, ICLTensor *output, const int16_t *conv, uint32_t scale, BorderMode border_mode, uint8_t constant_border_value)
{
    auto k = arm_compute::cpp14::make_unique<CLConvolution3x3Kernel>();
    k->configure(input, output, conv, scale, border_mode == BorderMode::UNDEFINED);
    _kernel = std::move(k);
    _border_handler.configure(input, _kernel->border_size(), border_mode, PixelValue(constant_border_value));
}

template <unsigned int matrix_size>
CLConvolutionSquare<matrix_size>::CLConvolutionSquare()
    : _tmp(), _is_separable(false), _kernel_hor(), _kernel_vert(), _kernel(), _border_handler()
{
}

template <unsigned int matrix_size>
void CLConvolutionSquare<matrix_size>::configure(ICLTensor *input, ICLTensor *output, const int16_t *conv, uint32_t scale, BorderMode border_mode, uint8_t constant_border_value)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(conv == nullptr);
    int16_t conv_col[matrix_size];
    int16_t conv_row[matrix_size];
    _is_separable = separate_matrix(conv, conv_col, conv_row, matrix_size);

    if(_is_separable)
    {
        std::pair<DataType, DataType> type_pair = data_type_for_convolution(conv_col, conv_row, matrix_size);
        _tmp.allocator()->init(TensorInfo(input->info()->tensor_shape(), 1, type_pair.first));

        if(scale == 0)
        {
            scale = calculate_matrix_scale(conv, matrix_size);
        }

        _kernel_hor.configure(input, &_tmp, conv_row, border_mode == BorderMode::UNDEFINED);
        _kernel_vert.configure(&_tmp, output, conv_col, scale, border_mode == BorderMode::UNDEFINED, type_pair.second);
        _border_handler.configure(input, _kernel_hor.border_size(), border_mode, PixelValue(constant_border_value));

        // Allocate intermediate buffer
        _tmp.allocator()->allocate();
    }
    else
    {
        _kernel.configure(input, output, conv, scale, border_mode == BorderMode::UNDEFINED);
        _border_handler.configure(input, _kernel.border_size(), border_mode, PixelValue(constant_border_value));
    }
}

template <unsigned int matrix_size>
void                   CLConvolutionSquare<matrix_size>::run()
{
    CLScheduler::get().enqueue(_border_handler);

    if(_is_separable)
    {
        CLScheduler::get().enqueue(_kernel_hor, false);
        CLScheduler::get().enqueue(_kernel_vert);
    }
    else
    {
        CLScheduler::get().enqueue(_kernel);
    }
}

template class arm_compute::CLConvolutionSquare<5>;
template class arm_compute::CLConvolutionSquare<7>;
template class arm_compute::CLConvolutionSquare<9>;

void CLConvolutionRectangle::configure(ICLTensor *input, ICLTensor *output, const int16_t *conv, uint32_t rows, uint32_t cols, uint32_t scale, BorderMode border_mode, uint8_t constant_border_value)
{
    auto k = arm_compute::cpp14::make_unique<CLConvolutionRectangleKernel>();
    k->configure(input, output, conv, rows, cols, scale, border_mode == BorderMode::UNDEFINED);
    _kernel = std::move(k);
    _border_handler.configure(input, _kernel->border_size(), border_mode, PixelValue(constant_border_value));
}
