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
#include "arm_compute/core/CL/kernels/CLGaussian5x5Kernel.h"

#include <cstdint>

using namespace arm_compute;

void CLGaussian5x5HorKernel::configure(const ICLTensor *input, ICLTensor *output, bool border_undefined)
{
    const int16_t matrix[] = { 1, 4, 6, 4, 1 };

    // Set arguments
    CLSeparableConvolution5x5HorKernel::configure(input, output, matrix, border_undefined);
}

void CLGaussian5x5VertKernel::configure(const ICLTensor *input, ICLTensor *output, bool border_undefined)
{
    const uint32_t scale    = 256;
    const int16_t  matrix[] = { 1, 4, 6, 4, 1 };

    // Set arguments
    CLSeparableConvolution5x5VertKernel::configure(input, output, matrix, scale, border_undefined);
}
