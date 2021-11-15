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
#include "arm_compute/runtime/NEON/functions/NEPixelWiseMultiplication.h"

#include "arm_compute/core/ITensor.h"
#include "src/cpu/operators/CpuMul.h"

#include <utility>

namespace arm_compute
{
struct NEPixelWiseMultiplication::Impl
{
    const ITensor               *src_0{ nullptr };
    const ITensor               *src_1{ nullptr };
    ITensor                     *dst{ nullptr };
    std::unique_ptr<cpu::CpuMul> op{ nullptr };
};

NEPixelWiseMultiplication::NEPixelWiseMultiplication()
    : _impl(std::make_unique<Impl>())
{
}
NEPixelWiseMultiplication::~NEPixelWiseMultiplication() = default;

Status NEPixelWiseMultiplication::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, float scale, ConvertPolicy overflow_policy, RoundingPolicy rounding_policy,
                                           const ActivationLayerInfo &act_info)
{
    return cpu::CpuMul::validate(input1, input2, output, scale, overflow_policy, rounding_policy, act_info);
}

void NEPixelWiseMultiplication::configure(const ITensor *input1, const ITensor *input2, ITensor *output, float scale, ConvertPolicy overflow_policy, RoundingPolicy rounding_policy,
                                          const ActivationLayerInfo &act_info)
{
    _impl->src_0 = input1;
    _impl->src_1 = input2;
    _impl->dst   = output;
    _impl->op    = std::make_unique<cpu::CpuMul>();
    _impl->op->configure(input1->info(), input2->info(), output->info(), scale, overflow_policy, rounding_policy, act_info);
}

void NEPixelWiseMultiplication::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src_0);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->src_1);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}

struct NEComplexPixelWiseMultiplication::Impl
{
    ITensor                            *src_0{ nullptr };
    ITensor                            *src_1{ nullptr };
    ITensor                            *dst{ nullptr };
    std::unique_ptr<cpu::CpuComplexMul> op{ nullptr };
};

NEComplexPixelWiseMultiplication::NEComplexPixelWiseMultiplication()
    : _impl(std::make_unique<Impl>())
{
}
NEComplexPixelWiseMultiplication::~NEComplexPixelWiseMultiplication() = default;

Status NEComplexPixelWiseMultiplication::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    return cpu::CpuComplexMul::validate(input1, input2, output, act_info);
}

void NEComplexPixelWiseMultiplication::configure(ITensor *input1, ITensor *input2, ITensor *output, const ActivationLayerInfo &act_info)
{
    _impl->src_0 = input1;
    _impl->src_1 = input2;
    _impl->dst   = output;
    _impl->op    = std::make_unique<cpu::CpuComplexMul>();
    _impl->op->configure(input1->info(), input2->info(), output->info(), act_info);
}

void NEComplexPixelWiseMultiplication::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src_0);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->src_1);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}
} // namespace arm_compute
