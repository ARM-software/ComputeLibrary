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
#include "arm_compute/runtime/NEON/functions/NEPoolingLayer.h"

#include "arm_compute/core/Validate.h"
#include "src/runtime/cpu/operators/CpuPooling.h"

namespace arm_compute
{
struct NEPoolingLayer::Impl
{
    ITensor                         *src{ nullptr };
    ITensor                         *dst{ nullptr };
    ITensor                         *indices{ nullptr };
    std::shared_ptr<IMemoryManager>  memory_manager{ nullptr };
    std::unique_ptr<cpu::CpuPooling> op{ nullptr };
};

NEPoolingLayer::~NEPoolingLayer() = default;

NEPoolingLayer::NEPoolingLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _impl(std::make_unique<Impl>())
{
    _impl->memory_manager = std::move(memory_manager);
}

void NEPoolingLayer::configure(ITensor *input, ITensor *output, const PoolingLayerInfo &pool_info, ITensor *indices)
{
    _impl->src     = input;
    _impl->dst     = output;
    _impl->indices = indices;
    _impl->op      = std::make_unique<cpu::CpuPooling>(_impl->memory_manager);
    _impl->op->configure(input->info(), output->info(), pool_info, (indices) ? indices->info() : nullptr);
}

Status NEPoolingLayer::validate(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &pool_info, const ITensorInfo *indices)
{
    return cpu::CpuPooling::validate(input, output, pool_info, indices);
}

void NEPoolingLayer::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST_0, _impl->dst);
    pack.add_tensor(TensorType::ACL_DST_1, _impl->indices);
    _impl->op->run(pack);
}
} // namespace arm_compute
