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
#include "arm_compute/runtime/NEON/functions/NEGEMMConv2d.h"

#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/runtime/cpu/operators/CpuGemmDirectConv2d.h"

#include <set>

namespace arm_compute
{
using OperatorType = cpu::CpuGemmDirectConv2d;

struct NEGEMMConv2d::Impl
{
    ITensorPack                   tensors{};
    std::unique_ptr<OperatorType> op{ nullptr };
};

NEGEMMConv2d::NEGEMMConv2d(const std::shared_ptr<IMemoryManager> &memory_manager)
    : _impl(std::make_unique<Impl>())
{
    _impl->op = std::make_unique<OperatorType>(memory_manager);
}

NEGEMMConv2d::~NEGEMMConv2d() = default;

void NEGEMMConv2d::configure(ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const Conv2dInfo &info)
{
    _impl->tensors.add_const_tensor(TensorType::ACL_SRC_0, input);
    _impl->tensors.add_const_tensor(TensorType::ACL_SRC_1, weights);
    _impl->tensors.add_const_tensor(TensorType::ACL_SRC_2, biases);
    _impl->tensors.add_tensor(TensorType::ACL_DST, output);

    _impl->op->configure(input->info(), weights->info(), biases->info(), output->info(), info);
}

Status NEGEMMConv2d::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const Conv2dInfo &info)
{
    return OperatorType::validate(input, weights, biases, output, info);
}
void NEGEMMConv2d::run()
{
    _impl->op->run(_impl->tensors);
}
void NEGEMMConv2d::prepare()
{
    _impl->op->prepare(_impl->tensors);
}
} // namespace arm_compute
