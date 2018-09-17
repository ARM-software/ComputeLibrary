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
#include "arm_compute/runtime/NEON/functions/NENonLinearFilter.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/NEON/kernels/NENonLinearFilterKernel.h"
#include "arm_compute/core/PixelValue.h"

#include <utility>

using namespace arm_compute;

void NENonLinearFilter::configure(ITensor *input, ITensor *output, NonLinearFilterFunction function, unsigned int mask_size, MatrixPattern pattern, const uint8_t *mask,
                                  BorderMode border_mode,
                                  uint8_t    constant_border_value)
{
    auto k = arm_compute::cpp14::make_unique<NENonLinearFilterKernel>();
    k->configure(input, output, function, mask_size, pattern, mask, border_mode == BorderMode::UNDEFINED);
    _kernel = std::move(k);
    _border_handler.configure(input, _kernel->border_size(), border_mode, PixelValue(constant_border_value));
}
