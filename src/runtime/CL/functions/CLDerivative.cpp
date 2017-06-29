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
#include "arm_compute/runtime/CL/functions/CLDerivative.h"

#include "arm_compute/core/CL/kernels/CLDerivativeKernel.h"
#include "arm_compute/core/PixelValue.h"
#include "support/ToolchainSupport.h"

#include <utility>

using namespace arm_compute;

void CLDerivative::configure(ICLTensor *input, ICLTensor *output_x, ICLTensor *output_y, BorderMode border_mode, uint8_t constant_border_value)
{
    auto k = arm_compute::support::cpp14::make_unique<CLDerivativeKernel>();
    k->configure(input, output_x, output_y, border_mode == BorderMode::UNDEFINED);
    _kernel = std::move(k);
    _border_handler.configure(input, BorderSize(1), border_mode, PixelValue(constant_border_value));
}
