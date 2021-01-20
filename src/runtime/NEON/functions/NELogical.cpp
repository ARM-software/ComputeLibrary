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
#include "arm_compute/runtime/NEON/functions/NELogical.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/Tensor.h"
#include "src/core/NEON/kernels/NELogicalKernel.h"

namespace arm_compute
{
struct LogicalArgs
{
    std::unique_ptr<kernels::NELogicalKernel> kernel{ nullptr };
    ITensorPack                               pack{};
};

struct NELogicalAnd::Impl : public LogicalArgs
{
};
NELogicalAnd::NELogicalAnd()
    : _impl(std::make_unique<Impl>())
{
}
NELogicalAnd &NELogicalAnd::operator=(NELogicalAnd &&) = default;
NELogicalAnd::~NELogicalAnd()                          = default;

void NELogicalAnd::configure(const ITensor *input1, const ITensor *input2, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, output);

    _impl->kernel = std::make_unique<kernels::NELogicalKernel>();
    _impl->kernel->configure(input1->info(), input2->info(), output->info(), kernels::LogicalOperation::And);

    _impl->pack = ITensorPack();
    _impl->pack.add_tensor(TensorType::ACL_SRC_0, input1);
    _impl->pack.add_tensor(TensorType::ACL_SRC_1, input2);
    _impl->pack.add_tensor(TensorType::ACL_DST, output);
}

Status NELogicalAnd::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output)
{
    return kernels::NELogicalKernel::validate(input1, input2, output, kernels::LogicalOperation::And);
}

void NELogicalAnd::run()
{
    NEScheduler::get().schedule_op(_impl->kernel.get(), Window::DimY, _impl->kernel->window(), _impl->pack);
}

struct NELogicalOr::Impl : public LogicalArgs
{
};
NELogicalOr::NELogicalOr()
    : _impl(std::make_unique<Impl>())
{
}
NELogicalOr &NELogicalOr::operator=(NELogicalOr &&) = default;
NELogicalOr::~NELogicalOr()                         = default;

void NELogicalOr::configure(const ITensor *input1, const ITensor *input2, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, output);

    _impl->kernel = std::make_unique<kernels::NELogicalKernel>();
    _impl->kernel->configure(input1->info(), input2->info(), output->info(), kernels::LogicalOperation::Or);

    _impl->pack = ITensorPack();
    _impl->pack.add_tensor(TensorType::ACL_SRC_0, input1);
    _impl->pack.add_tensor(TensorType::ACL_SRC_1, input2);
    _impl->pack.add_tensor(TensorType::ACL_DST, output);
}

Status NELogicalOr::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output)
{
    return kernels::NELogicalKernel::validate(input1, input2, output, kernels::LogicalOperation::Or);
}

void NELogicalOr::run()
{
    NEScheduler::get().schedule_op(_impl->kernel.get(), Window::DimY, _impl->kernel->window(), _impl->pack);
}

struct NELogicalNot::Impl : public LogicalArgs
{
};
NELogicalNot::NELogicalNot()
    : _impl(std::make_unique<Impl>())
{
}
NELogicalNot &NELogicalNot::operator=(NELogicalNot &&) = default;
NELogicalNot::~NELogicalNot()                          = default;

void NELogicalNot::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    _impl->kernel = std::make_unique<kernels::NELogicalKernel>();
    _impl->kernel->configure(input->info(), nullptr, output->info(), kernels::LogicalOperation::Not);

    _impl->pack = ITensorPack();
    _impl->pack.add_tensor(TensorType::ACL_SRC_0, input);
    _impl->pack.add_tensor(TensorType::ACL_DST, output);
}

Status NELogicalNot::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return kernels::NELogicalKernel::validate(input, nullptr, output, kernels::LogicalOperation::Not);
}

void NELogicalNot::run()
{
    NEScheduler::get().schedule_op(_impl->kernel.get(), Window::DimY, _impl->kernel->window(), _impl->pack);
}
} // namespace arm_compute
