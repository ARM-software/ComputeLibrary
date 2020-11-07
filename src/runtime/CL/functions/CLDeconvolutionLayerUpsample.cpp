/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLDeconvolutionLayerUpsample.h"

#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "src/core/CL/kernels/CLDeconvolutionLayerUpsampleKernel.h"
#include "src/core/CL/kernels/CLMemsetKernel.h"
#include "support/MemorySupport.h"

namespace arm_compute
{
CLDeconvolutionLayerUpsample::CLDeconvolutionLayerUpsample() // NOLINT
    : _upsample(support::cpp14::make_unique<CLDeconvolutionLayerUpsampleKernel>()),
      _memset(support::cpp14::make_unique<CLMemsetKernel>()),
      _output(nullptr)
{
}

CLDeconvolutionLayerUpsample::~CLDeconvolutionLayerUpsample() = default;

Status CLDeconvolutionLayerUpsample::validate(const ITensorInfo *input, const ITensorInfo *output, const PadStrideInfo &info)
{
    return CLDeconvolutionLayerUpsampleKernel::validate(input, output, info);
}

void CLDeconvolutionLayerUpsample::configure(ICLTensor *input, ICLTensor *output, const PadStrideInfo &info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, info);
}

void CLDeconvolutionLayerUpsample::configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output, const PadStrideInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    _output = output;
    _memset->configure(compile_context, _output, PixelValue(0, _output->info()->data_type(), _output->info()->quantization_info()));
    _upsample->configure(compile_context, input, _output, info);
}

void CLDeconvolutionLayerUpsample::run()
{
    CLScheduler::get().enqueue(*_memset, false);
    CLScheduler::get().enqueue(*_upsample, true);
}
} // namespace arm_compute
