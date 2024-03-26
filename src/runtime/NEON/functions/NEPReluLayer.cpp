/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEPReluLayer.h"

#include "arm_compute/core/ITensor.h"

#include "src/cpu/operators/CpuPRelu.h"

namespace arm_compute
{
using OperatorType = cpu::CpuPRelu;

struct NEPReluLayer::Impl
{
    const ITensor                *src_0{nullptr};
    const ITensor                *src_1{nullptr};
    ITensor                      *dst{nullptr};
    std::unique_ptr<OperatorType> op{nullptr};
};

NEPReluLayer::NEPReluLayer() : _impl(std::make_unique<Impl>())
{
}
NEPReluLayer::NEPReluLayer(NEPReluLayer &&)            = default;
NEPReluLayer &NEPReluLayer::operator=(NEPReluLayer &&) = default;
NEPReluLayer::~NEPReluLayer()                          = default;

void NEPReluLayer::configure(const ITensor *input, const ITensor *alpha, ITensor *output)
{
    _impl->src_0 = input;
    _impl->src_1 = alpha;
    _impl->dst   = output;
    _impl->op    = std::make_unique<OperatorType>();
    _impl->op->configure(input->info(), alpha->info(), output->info());
}

void NEPReluLayer::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src_0);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->src_1);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}

Status NEPReluLayer::validate(const ITensorInfo *input, const ITensorInfo *alpha, const ITensorInfo *output)
{
    return OperatorType::validate(input, alpha, output);
}
} // namespace arm_compute
