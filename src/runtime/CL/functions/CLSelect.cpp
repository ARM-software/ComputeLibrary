/*
 * Copyright (c) 2018-2021, 2024 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLSelect.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include "src/common/utils/Log.h"
#include "src/core/CL/kernels/CLSelectKernel.h"

using namespace arm_compute;

namespace arm_compute
{
void CLSelect::configure(const ICLTensor *c, const ICLTensor *x, const ICLTensor *y, ICLTensor *output)
{
    configure(CLKernelLibrary::get().get_compile_context(), c, x, y, output);
}

void CLSelect::configure(const CLCompileContext &compile_context,
                         const ICLTensor        *c,
                         const ICLTensor        *x,
                         const ICLTensor        *y,
                         ICLTensor              *output)
{
    ARM_COMPUTE_LOG_PARAMS(c, x, y, output);
    auto k = std::make_unique<CLSelectKernel>();
    k->configure(compile_context, c, x, y, output);
    _kernel = std::move(k);
}

Status CLSelect::validate(const ITensorInfo *c, const ITensorInfo *x, const ITensorInfo *y, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DYNAMIC_SHAPE(c, x, y, output);
    return CLSelectKernel::validate(c, x, y, output);
}
} // namespace arm_compute
