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
#include "arm_compute/runtime/NEON/functions/NEFill.h"

#include "arm_compute/core/Validate.h"

#include "src/cpu/operators/CpuFill.h"

#include <utility>

namespace arm_compute
{
struct NEFill::Impl
{
    ITensor                      *tensor{nullptr};
    std::unique_ptr<cpu::CpuFill> op{nullptr};
};

NEFill::NEFill() : _impl(std::make_unique<Impl>())
{
}
NEFill::NEFill(NEFill &&)            = default;
NEFill &NEFill::operator=(NEFill &&) = default;
NEFill::~NEFill()                    = default;

void NEFill::configure(ITensor *tensor, PixelValue constant_value)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(tensor);

    _impl->tensor = tensor;
    _impl->op     = std::make_unique<cpu::CpuFill>();
    _impl->op->configure(tensor->info(), constant_value);
}

void NEFill::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_DST, _impl->tensor);
    _impl->op->run(pack);
}
} // namespace arm_compute
