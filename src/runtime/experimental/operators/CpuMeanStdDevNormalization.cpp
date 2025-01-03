/*
 * Copyright (c) 2019-2025 Arm Limited.
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

#include "src/cpu/operators/CpuMeanStdDevNormalization.h"

#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/experimental/operators/CpuMeanStdDevNormalization.h"

namespace arm_compute
{
namespace experimental
{
namespace op
{
struct CpuMeanStdDevNormalization::Impl
{
    std::unique_ptr<cpu::CpuMeanStdDevNormalization> op{nullptr};
};

CpuMeanStdDevNormalization::CpuMeanStdDevNormalization() : impl_(std::make_unique<Impl>())
{
}
CpuMeanStdDevNormalization::~CpuMeanStdDevNormalization() = default;

void CpuMeanStdDevNormalization::configure(ITensorInfo *input, ITensorInfo *output, float epsilon)
{
    impl_->op = std::make_unique<cpu::CpuMeanStdDevNormalization>();
    impl_->op->configure(input, output, epsilon);
}

Status CpuMeanStdDevNormalization::validate(const ITensorInfo *input, const ITensorInfo *output, float epsilon)
{
    return cpu::CpuMeanStdDevNormalization::validate(input, output, epsilon);
}

void CpuMeanStdDevNormalization::run(ITensorPack &tensors)
{
    impl_->op->run(tensors);
}

} // namespace op
} // namespace experimental
} // namespace arm_compute
