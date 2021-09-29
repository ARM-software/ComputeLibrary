/*
 * Copyright (c) 2018-2021 Arm Limited.
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

#include "arm_compute/runtime/CL/functions/CLPriorBoxLayer.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "src/core/CL/kernels/CLFillBorderKernel.h"
#include "src/core/CL/kernels/CLPriorBoxLayerKernel.h"

#include "src/common/utils/Log.h"

using namespace arm_compute;

CLPriorBoxLayer::CLPriorBoxLayer()
    : _min(nullptr), _max(nullptr), _aspect_ratios(nullptr)
{
}

void CLPriorBoxLayer::configure(const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, const PriorBoxLayerInfo &info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input1, input2, output, info);
}

void CLPriorBoxLayer::configure(const CLCompileContext &compile_context, const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, const PriorBoxLayerInfo &info)
{
    ARM_COMPUTE_LOG_PARAMS(input1, input2, output, info);
    _min           = cl::Buffer(CLScheduler::get().context(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, info.min_sizes().size() * sizeof(float));
    _aspect_ratios = cl::Buffer(CLScheduler::get().context(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, info.aspect_ratios().size() * sizeof(float));
    if(!info.max_sizes().empty())
    {
        _max = cl::Buffer(CLScheduler::get().context(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, info.max_sizes().size() * sizeof(float));
    }

    auto k = std::make_unique<CLPriorBoxLayerKernel>();
    k->configure(compile_context, input1, input2, output, info, &_min, &_max, &_aspect_ratios);
    _kernel = std::move(k);
}

Status CLPriorBoxLayer::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const PriorBoxLayerInfo &info)
{
    return CLPriorBoxLayerKernel::validate(input1, input2, output, info);
}
