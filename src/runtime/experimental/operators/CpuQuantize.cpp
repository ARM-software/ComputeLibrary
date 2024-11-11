/*
 * Copyright (c) 2017-2021, 2024 Arm Limited.
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

#include "src/cpu/operators/CpuQuantize.h"

#include "arm_compute/runtime/experimental/operators/CpuQuantize.h"

namespace arm_compute
{
namespace experimental
{
namespace op
{
struct CpuQuantize::Impl
{
    std::unique_ptr<cpu::CpuQuantize> op{nullptr};
};

CpuQuantize::CpuQuantize() : impl_(std::make_unique<Impl>())
{
}
CpuQuantize::~CpuQuantize() = default;

Status CpuQuantize::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return cpu::CpuQuantize::validate(input, output);
}

void CpuQuantize::configure(const ITensorInfo *input, ITensorInfo *output)
{
    impl_->op = std::make_unique<cpu::CpuQuantize>();
    impl_->op->configure(input, output);
}

void CpuQuantize::run(ITensorPack &pack)
{
    impl_->op->run(pack);
}
} // namespace op
} // namespace experimental
} // namespace arm_compute
