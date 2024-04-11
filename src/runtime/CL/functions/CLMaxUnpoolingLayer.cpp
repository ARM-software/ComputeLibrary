/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLMaxUnpoolingLayer.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include "src/common/utils/Log.h"
#include "src/core/CL/kernels/CLMaxUnpoolingLayerKernel.h"

namespace arm_compute
{
CLMaxUnpoolingLayer::CLMaxUnpoolingLayer()
    : _fill(), _unpooling_layer_kernel(std::make_unique<CLMaxUnpoolingLayerKernel>())
{
}

CLMaxUnpoolingLayer::~CLMaxUnpoolingLayer() = default;

void CLMaxUnpoolingLayer::configure(ICLTensor              *input,
                                    ICLTensor              *indices,
                                    ICLTensor              *output,
                                    const PoolingLayerInfo &pool_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, indices, output, pool_info);
}

void CLMaxUnpoolingLayer::configure(const CLCompileContext &compile_context,
                                    ICLTensor              *input,
                                    ICLTensor              *indices,
                                    ICLTensor              *output,
                                    const PoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_LOG_PARAMS(input, indices, output, pool_info);
    const PixelValue zero_value(0.f);
    _fill.configure(output, zero_value);

    _unpooling_layer_kernel->configure(compile_context, input, indices, output, pool_info);
}

Status CLMaxUnpoolingLayer::validate(const ITensorInfo      *input,
                                     const ITensorInfo      *indices,
                                     const ITensorInfo      *output,
                                     const PoolingLayerInfo &pool_info)
{
    return CLMaxUnpoolingLayerKernel::validate(input, indices, output, pool_info);
}

void CLMaxUnpoolingLayer::run()
{
    // Run fill
    _fill.run();

    // Run max unpooling layer
    CLScheduler::get().enqueue(*_unpooling_layer_kernel);
}
} /* namespace arm_compute */
