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
#include "arm_compute/runtime/NEON/functions/NEConvolution.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "src/core/NEON/kernels/NEConvolutionKernel.h"
#include "src/core/NEON/kernels/NEConvolutionKernel.h"
#include "src/core/NEON/kernels/NEFillBorderKernel.h"

#include <array>
#include <utility>

namespace arm_compute
{
NEConvolution3x3::~NEConvolution3x3() = default;

void NEConvolution3x3::configure(ITensor *input, ITensor *output, const int16_t *conv, uint32_t scale, BorderMode border_mode, uint8_t constant_border_value)
{
    auto k = std::make_unique<NEConvolution3x3Kernel>();
    k->configure(input, output, conv, scale, border_mode == BorderMode::UNDEFINED);
    _kernel = std::move(k);

    auto b = std::make_unique<NEFillBorderKernel>();
    b->configure(input, _kernel->border_size(), border_mode, PixelValue(constant_border_value));
    _border_handler = std::move(b);
}

template <unsigned int matrix_size>
NEConvolutionSquare<matrix_size>::~NEConvolutionSquare() = default;

template <unsigned int matrix_size>
NEConvolutionSquare<matrix_size>::NEConvolutionSquare(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _tmp(), _is_separable(false), _kernel_hor(), _kernel_vert(), _kernel(), _border_handler()
{
}

template <unsigned int matrix_size>
void NEConvolutionSquare<matrix_size>::configure(ITensor *input, ITensor *output, const int16_t *conv, uint32_t scale, BorderMode border_mode,
                                                 uint8_t constant_border_value)
{
    ARM_COMPUTE_ERROR_ON(conv == nullptr);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8, DataType::S16);

    std::array<int16_t, matrix_size> conv_col{ { 0 } };
    std::array<int16_t, matrix_size> conv_row{ { 0 } };

    _is_separable = separate_matrix(conv, conv_col.data(), conv_row.data(), matrix_size);

    auto b = std::make_unique<NEFillBorderKernel>();
    if(_is_separable)
    {
        DataType intermediate_type = DataType::UNKNOWN;
        std::tie(std::ignore, intermediate_type) = data_type_for_convolution(conv_col.data(), conv_row.data(), matrix_size);

        _tmp.allocator()->init(TensorInfo(input->info()->tensor_shape(), 1, intermediate_type));

        // Manage intermediate buffers
        _memory_group.manage(&_tmp);

        // Calculate scale
        if(scale == 0)
        {
            scale = calculate_matrix_scale(conv, matrix_size);
        }

        _kernel_hor  = std::make_unique<NESeparableConvolutionHorKernel<matrix_size>>();
        _kernel_vert = std::make_unique<NESeparableConvolutionVertKernel<matrix_size>>();

        _kernel_hor->configure(input, &_tmp, conv_row.data(), border_mode == BorderMode::UNDEFINED);
        _kernel_vert->configure(&_tmp, output, conv_col.data(), scale, border_mode == BorderMode::UNDEFINED);

        _tmp.allocator()->allocate();

        b->configure(input, _kernel_hor->border_size(), border_mode, PixelValue(constant_border_value));
    }
    else
    {
        _kernel = std::make_unique<NEConvolutionKernel<matrix_size>>();
        _kernel->configure(input, output, conv, scale, border_mode == BorderMode::UNDEFINED);
        b->configure(input, _kernel->border_size(), border_mode, PixelValue(constant_border_value));
    }
    _border_handler = std::move(b);
}

template <unsigned int matrix_size>
void                   NEConvolutionSquare<matrix_size>::run()
{
    NEScheduler::get().schedule(_border_handler.get(), Window::DimZ);

    if(_is_separable)
    {
        MemoryGroupResourceScope scope_mg(_memory_group);

        NEScheduler::get().schedule(_kernel_hor.get(), Window::DimY);
        NEScheduler::get().schedule(_kernel_vert.get(), Window::DimY);
    }
    else
    {
        NEScheduler::get().schedule(_kernel.get(), Window::DimY);
    }
}

template class arm_compute::NEConvolutionSquare<5>;
template class arm_compute::NEConvolutionSquare<7>;
template class arm_compute::NEConvolutionSquare<9>;

NEConvolutionRectangle::~NEConvolutionRectangle() = default;

void NEConvolutionRectangle::configure(ITensor *input, ITensor *output, const int16_t *conv, uint32_t rows, uint32_t cols, uint32_t scale, BorderMode border_mode, uint8_t constant_border_value)
{
    border_mode = (border_mode == BorderMode::UNDEFINED) ? BorderMode::CONSTANT : border_mode;
    auto k      = std::make_unique<NEConvolutionRectangleKernel>();
    k->configure(input, output, conv, rows, cols, scale, false);
    _kernel = std::move(k);

    auto b = std::make_unique<NEFillBorderKernel>();
    b->configure(input, _kernel->border_size(), border_mode, PixelValue(constant_border_value));
    _border_handler = std::move(b);
}
} // namespace arm_compute
