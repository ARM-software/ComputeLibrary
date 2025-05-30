/*
 * Copyright (c) 2025 Arm Limited.
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

#include "arm_compute/runtime/experimental/operators/CpuPool2d.h"

#include "src/cpu/operators/CpuPool2d.h"

namespace arm_compute
{
namespace experimental
{
namespace op
{

struct CpuPool2d::Impl
{
    std::unique_ptr<arm_compute::cpu::CpuPool2d> op{nullptr};
};

CpuPool2d::CpuPool2d() : impl_(std::make_unique<Impl>())
{
    impl_->op = std::make_unique<cpu::CpuPool2d>();
}

CpuPool2d::~CpuPool2d() = default;

void CpuPool2d::configure(ITensorInfo *src, ITensorInfo *dst, const PoolingLayerInfo &pool_info, ITensorInfo *indices)
{
    impl_->op->configure(src, dst, pool_info, indices);
}

Status CpuPool2d::validate(const ITensorInfo      *src,
                           const ITensorInfo      *dst,
                           const PoolingLayerInfo &pool_info,
                           const ITensorInfo      *indices)
{
    return cpu::CpuPool2d::validate(src, dst, pool_info, indices);
}

void CpuPool2d::run(ITensorPack &tensors)
{
    impl_->op->run(tensors);
}

experimental::MemoryRequirements CpuPool2d::workspace() const
{
    return impl_->op->workspace();
}

} //namespace op
} //namespace experimental
} //namespace arm_compute
