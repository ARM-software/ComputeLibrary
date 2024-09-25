/*
 * Copyright (c) 2021, 2023-2024 Arm Limited.
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

#include "arm_compute/runtime/experimental/operators/CpuSoftmax.h"

#include "src/cpu/operators/CpuSoftmax.h"

namespace arm_compute
{
namespace experimental
{
namespace op
{

struct CpuSoftmax::Impl
{
    std::unique_ptr<arm_compute::cpu::CpuSoftmaxGeneric> op{nullptr};
};

CpuSoftmax::CpuSoftmax() : impl_(std::make_unique<Impl>())
{
    impl_->op = std::make_unique<cpu::CpuSoftmaxGeneric>();
}

CpuSoftmax::~CpuSoftmax() = default;

void CpuSoftmax::configure(const ITensorInfo *src, ITensorInfo *dst, float beta, int32_t axis, bool is_log)
{
    impl_->op->configure(src, dst, beta, axis, is_log);
}

Status CpuSoftmax::validate(const ITensorInfo *src, const ITensorInfo *dst, float beta, int32_t axis, bool is_log)
{
    return cpu::CpuSoftmaxGeneric::validate(src, dst, beta, axis, is_log);
}

void CpuSoftmax::run(ITensorPack &tensor)
{
    impl_->op->run(tensor);
}

experimental::MemoryRequirements CpuSoftmax::workspace() const
{
    return impl_->op->workspace();
}

void CpuSoftmax::prepare(ITensorPack &constants)
{
    ARM_COMPUTE_UNUSED(constants);
}

} // namespace op
} // namespace experimental
} // namespace arm_compute
