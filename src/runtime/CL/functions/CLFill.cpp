/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLFill.h"

#include "arm_compute/core/Types.h"
#include "src/core/CL/kernels/CLMemsetKernel.h"

#include "support/MemorySupport.h"

#include <utility>

namespace arm_compute
{
void CLFill::configure(ICLTensor *tensor, PixelValue constant_value)
{
    configure(CLKernelLibrary::get().get_compile_context(), tensor, constant_value);
}

void CLFill::configure(const CLCompileContext &compile_context, ICLTensor *tensor, PixelValue constant_value)
{
    auto k = arm_compute::support::cpp14::make_unique<CLMemsetKernel>();
    k->configure(compile_context, tensor, constant_value);
    _kernel = std::move(k);
}
} // namespace arm_compute
