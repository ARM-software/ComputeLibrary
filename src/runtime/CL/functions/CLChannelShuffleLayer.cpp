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
#include "arm_compute/runtime/CL/functions/CLChannelShuffleLayer.h"

#include "arm_compute/core/Types.h"

#include "src/common/utils/Log.h"
#include "src/core/CL/kernels/CLChannelShuffleLayerKernel.h"

namespace arm_compute
{
void CLChannelShuffleLayer::configure(const ICLTensor *input, ICLTensor *output, unsigned int num_groups)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, num_groups);
}

void CLChannelShuffleLayer::configure(const CLCompileContext &compile_context,
                                      const ICLTensor        *input,
                                      ICLTensor              *output,
                                      unsigned int            num_groups)
{
    ARM_COMPUTE_LOG_PARAMS(input, output, num_groups);
    auto k = std::make_unique<CLChannelShuffleLayerKernel>();
    k->configure(compile_context, input, output, num_groups);
    _kernel = std::move(k);
}

Status CLChannelShuffleLayer::validate(const ITensorInfo *input, const ITensorInfo *output, unsigned int num_groups)
{
    return CLChannelShuffleLayerKernel::validate(input, output, num_groups);
}
} // namespace arm_compute
