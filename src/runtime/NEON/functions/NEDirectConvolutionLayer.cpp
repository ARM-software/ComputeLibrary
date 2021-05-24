/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEDirectConvolutionLayer.h"

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/runtime/cpu/operators/CpuDirectConv2d.h"

namespace arm_compute
{
struct NEDirectConvolutionLayer::Impl
{
    ITensor                              *src{ nullptr };
    const ITensor                        *weights{ nullptr };
    const ITensor                        *bias{ nullptr };
    ITensor                              *dst{ nullptr };
    std::unique_ptr<cpu::CpuDirectConv2d> op{ nullptr };
};

NEDirectConvolutionLayer::NEDirectConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_manager(std::move(memory_manager)), _impl(std::make_unique<Impl>())
{
}
NEDirectConvolutionLayer::~NEDirectConvolutionLayer() = default;

void NEDirectConvolutionLayer::configure(ITensor *input, const ITensor *weights, const ITensor *bias, ITensor *output, const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info)
{
    _impl->src     = input;
    _impl->weights = weights;
    _impl->bias    = bias;
    _impl->dst     = output;
    _impl->op      = std::make_unique<cpu::CpuDirectConv2d>(_memory_manager);
    _impl->op->configure(input->info(), weights->info(), (bias != nullptr ? bias->info() : nullptr), output->info(), conv_info, act_info);
}

Status NEDirectConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *bias, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                          const ActivationLayerInfo &act_info)
{
    return cpu::CpuDirectConv2d::validate(input, weights, bias, output, conv_info, act_info);
}

void NEDirectConvolutionLayer::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->weights);
    pack.add_tensor(TensorType::ACL_SRC_2, _impl->bias);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}
} // namespace arm_compute
