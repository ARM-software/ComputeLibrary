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
#include "arm_compute/runtime/CL/functions/CLIntegralImage.h"

#include "arm_compute/runtime/CL/CLScheduler.h"
#include "src/core/CL/kernels/CLIntegralImageKernel.h"
#include "support/MemorySupport.h"

using namespace arm_compute;

CLIntegralImage::CLIntegralImage()
    : _integral_hor(support::cpp14::make_unique<CLIntegralImageHorKernel>()),
      _integral_vert(support::cpp14::make_unique<CLIntegralImageVertKernel>())
{
}

CLIntegralImage::~CLIntegralImage() = default;

void CLIntegralImage::configure(const ICLTensor *input, ICLTensor *output)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output);
}

void CLIntegralImage::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output)
{
    _integral_hor->configure(compile_context, input, output);
    _integral_vert->configure(compile_context, output);
}

void CLIntegralImage::run()
{
    CLScheduler::get().enqueue(*_integral_hor, false);
    CLScheduler::get().enqueue(*_integral_vert);
}
