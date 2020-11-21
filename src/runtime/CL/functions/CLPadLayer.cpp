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
#include "arm_compute/runtime/CL/functions/CLPadLayer.h"
#include "src/core/CL/kernels/CLCopyKernel.h"
#include "src/core/CL/kernels/CLPadLayerKernel.h"

namespace arm_compute
{
CLPadLayer::CLPadLayer()
    : _pad_kernel(std::make_unique<CLPadLayerKernel>()),
      _copy_kernel(std::make_unique<CLCopyKernel>()),
      _perform_pad(false)
{
}

CLPadLayer::~CLPadLayer() = default;

void CLPadLayer::configure(ICLTensor *input, ICLTensor *output, const PaddingList &padding, PixelValue constant_value, PaddingMode mode)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, padding, constant_value, mode);
}

void CLPadLayer::configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output, const PaddingList &padding, PixelValue constant_value, PaddingMode mode)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate(input->info(), output->info(), padding, constant_value, mode));

    _perform_pad = std::any_of(padding.begin(), padding.end(), [](PaddingInfo info)
    {
        return info.first > 0 || info.second > 0;
    });

    if(_perform_pad)
    {
        _pad_kernel->configure(compile_context, input, output, padding, constant_value, mode);
    }
    else
    {
        // Copy the input to the whole output if no padding is applied
        _copy_kernel->configure(compile_context, input, output);
    }
}
Status CLPadLayer::validate(const ITensorInfo *input, const ITensorInfo *output, const PaddingList &padding, PixelValue constant_value, PaddingMode mode)
{
    bool perform_pad = std::any_of(padding.begin(), padding.end(), [](PaddingInfo info)
    {
        return info.first > 0 || info.second > 0;
    });

    if(perform_pad)
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLPadLayerKernel::validate(input, output, padding, constant_value, mode));
    }
    else
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLCopyKernel::validate(input, output));
    }
    return Status{};
}
void CLPadLayer::run()
{
    if(_perform_pad)
    {
        CLScheduler::get().enqueue(*_pad_kernel);
    }
    else
    {
        CLScheduler::get().enqueue(*_copy_kernel);
    }
}
} // namespace arm_compute