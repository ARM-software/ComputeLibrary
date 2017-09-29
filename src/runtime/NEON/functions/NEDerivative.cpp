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
#include "arm_compute/runtime/NEON/functions/NEDerivative.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/NEON/kernels/NEDerivativeKernel.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

using namespace arm_compute;

NEDerivative::NEDerivative()
    : _kernel(), _border_handler()
{
}

void NEDerivative::configure(ITensor *input, ITensor *output_x, ITensor *output_y, BorderMode border_mode, uint8_t constant_border_value)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON((output_x == nullptr) && (output_y == nullptr));

    _kernel.configure(input, output_x, output_y, border_mode == BorderMode::UNDEFINED);
    _border_handler.configure(input, BorderSize(1), border_mode, PixelValue(constant_border_value));
}

void NEDerivative::run()
{
    NEScheduler::get().schedule(&_border_handler, Window::DimZ);
    NEScheduler::get().schedule(&_kernel, Window::DimY);
}
