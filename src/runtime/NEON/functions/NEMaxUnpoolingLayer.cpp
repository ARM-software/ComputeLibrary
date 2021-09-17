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
#include "arm_compute/runtime/NEON/functions/NEMaxUnpoolingLayer.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/NEON/functions/NEFill.h"
#include "src/common/utils/Log.h"
#include "src/core/NEON/kernels/NEMaxUnpoolingLayerKernel.h"

namespace arm_compute
{
NEMaxUnpoolingLayer::~NEMaxUnpoolingLayer() = default;

NEMaxUnpoolingLayer::NEMaxUnpoolingLayer()
    : _fill_func(), _unpooling_layer_kernel()
{
}

void NEMaxUnpoolingLayer::configure(ITensor *input, ITensor *indices, ITensor *output, const PoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_LOG_PARAMS(input, indices, output, pool_info);

    const PixelValue zero_value(0.f);
    _fill_func              = std::make_unique<NEFill>();
    _unpooling_layer_kernel = std::make_unique<NEMaxUnpoolingLayerKernel>();
    _fill_func->configure(output, zero_value);
    _unpooling_layer_kernel->configure(input, indices, output, pool_info);
}

Status NEMaxUnpoolingLayer::validate(const ITensorInfo *input, const ITensorInfo *indices, const ITensorInfo *output, const PoolingLayerInfo &pool_info)
{
    return NEMaxUnpoolingLayerKernel::validate(input, indices, output, pool_info);
}

void NEMaxUnpoolingLayer::run()
{
    _fill_func->run();
    NEScheduler::get().schedule(_unpooling_layer_kernel.get(), Window::DimY);
}
} /* namespace arm_compute */
