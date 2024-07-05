/*
 * Copyright (c) 2024 Arm Limited.
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
#include "arm_compute/runtime/experimental/operators/CpuElementwise.h"

#include "src/common/utils/Log.h"
#include "src/cpu/operators/CpuElementwise.h"

#include <memory>

namespace arm_compute
{
namespace experimental
{
namespace op
{

// CpuElementwiseDivision implementation:
struct CpuElementwiseDivision::Impl
{
    std::unique_ptr<cpu::CpuElementwiseDivision> divOp{nullptr};
};

CpuElementwiseDivision::CpuElementwiseDivision() : _impl(std::make_unique<Impl>())
{
    _impl->divOp = std::make_unique<cpu::CpuElementwiseDivision>();
}

CpuElementwiseDivision::~CpuElementwiseDivision() = default;

void CpuElementwiseDivision::configure(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst)
{
    _impl->divOp->configure(src0, src1, dst);
}

Status CpuElementwiseDivision::validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst)
{
    return cpu::CpuElementwiseDivision::validate(src0, src1, dst);
}

void CpuElementwiseDivision::run(ITensorPack &tensors)
{
    _impl->divOp->run(tensors);
}

// CpuElementwiseMax implementation:
struct CpuElementwiseMax::Impl
{
    std::unique_ptr<cpu::CpuElementwiseMax> maxOp{nullptr};
};

CpuElementwiseMax::CpuElementwiseMax() : _impl(std::make_unique<Impl>())
{
    _impl->maxOp = std::make_unique<cpu::CpuElementwiseMax>();
}

CpuElementwiseMax::~CpuElementwiseMax() = default;

void CpuElementwiseMax::configure(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst)
{
    _impl->maxOp->configure(src0, src1, dst);
}

Status CpuElementwiseMax::validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst)
{
    return cpu::CpuElementwiseMax::validate(src0, src1, dst);
}

void CpuElementwiseMax::run(ITensorPack &tensors)
{
    _impl->maxOp->run(tensors);
}

// CpuElementwiseMin implementation:
struct CpuElementwiseMin::Impl
{
    std::unique_ptr<cpu::CpuElementwiseMin> minOp{nullptr};
};

CpuElementwiseMin::CpuElementwiseMin() : _impl(std::make_unique<Impl>())
{
    _impl->minOp = std::make_unique<cpu::CpuElementwiseMin>();
}

CpuElementwiseMin::~CpuElementwiseMin() = default;

void CpuElementwiseMin::configure(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst)
{
    _impl->minOp->configure(src0, src1, dst);
}

Status CpuElementwiseMin::validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst)
{
    return cpu::CpuElementwiseMin::validate(src0, src1, dst);
}

void CpuElementwiseMin::run(ITensorPack &tensors)
{
    _impl->minOp->run(tensors);
}

} // namespace op
} // namespace experimental
} // namespace arm_compute
