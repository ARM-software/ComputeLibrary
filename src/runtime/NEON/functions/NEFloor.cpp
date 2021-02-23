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
#include "arm_compute/runtime/NEON/functions/NEFloor.h"

#include "arm_compute/core/Validate.h"
#include "src/runtime/cpu/operators/CpuFloor.h"

namespace arm_compute
{
struct NEFloor::Impl
{
    const ITensor                 *src{ nullptr };
    ITensor                       *dst{ nullptr };
    std::unique_ptr<cpu::CpuFloor> op{ nullptr };
};

NEFloor::NEFloor()
    : _impl(std::make_unique<Impl>())
{
}
NEFloor::NEFloor(NEFloor &&) = default;
NEFloor &NEFloor::operator=(NEFloor &&) = default;
NEFloor::~NEFloor()                     = default;

void NEFloor::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    _impl->src = input;
    _impl->dst = output;

    _impl->op = std::make_unique<cpu::CpuFloor>();
    _impl->op->configure(_impl->src->info(), _impl->dst->info());
}

Status NEFloor::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return cpu::CpuFloor::validate(input, output);
}

void NEFloor::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}
} // namespace arm_compute
