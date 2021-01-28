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
#include "arm_compute/runtime/CL/functions/CLScale.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

namespace arm_compute
{
CLScale::CLScale()
    : _border_handler(std::make_unique<CLFillBorderKernel>()), _kernel()
{
}

void CLScale::configure(ICLTensor *input, ICLTensor *output, const ScaleKernelInfo &info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, info);
}

void CLScale::configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output, const ScaleKernelInfo &info)
{
    auto k = std::make_unique<CLScaleKernel>();
    k->set_target(CLScheduler::get().target());
    k->configure(compile_context, input, output, info);
    _kernel = std::move(k);

    // Tune kernels
    CLScheduler::get().tune_kernel_static(*_kernel);

    const DataLayout data_layout = info.data_layout == DataLayout::UNKNOWN ? input->info()->data_layout() : info.data_layout;
    if(data_layout == DataLayout::NCHW && !_kernel->border_size().empty())
    {
        _border_handler->configure(compile_context, input, _kernel->border_size(), info.border_mode, info.constant_border_value);
    }
}

Status CLScale::validate(const ITensorInfo *input, const ITensorInfo *output, const ScaleKernelInfo &info)
{
    return CLScaleKernel::validate(input, output, info);
}

void CLScale::run()
{
    if(!_kernel->border_size().empty())
    {
        CLScheduler::get().enqueue(*_border_handler, false);
    }
    CLScheduler::get().enqueue(*_kernel);
}

} // namespace arm_compute
