/*
 * Copyright (c) 2016-2020 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEFillBorder.h"

#include "arm_compute/core/Window.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/core/NEON/kernels/NEFillBorderKernel.h"
#include "support/MemorySupport.h"

namespace arm_compute
{
void NEFillBorder::configure(ITensor *input, unsigned int border_width, BorderMode border_mode, const PixelValue &constant_border_value)
{
    _border_handler = arm_compute::support::cpp14::make_unique<NEFillBorderKernel>();
    _border_handler->configure(input, BorderSize(border_width), border_mode, constant_border_value);
}

void NEFillBorder::run()
{
    NEScheduler::get().schedule(_border_handler.get(), Window::DimZ);
}
} // namespace arm_compute