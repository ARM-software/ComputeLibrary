/*
 * Copyright (c) 2021, 2023 Arm Limited.
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
#include "src/cpu/operators/CpuFlatten.h"

#include "src/cpu/operators/CpuReshape.h"

#include "src/common/utils/Log.h"

namespace arm_compute
{
namespace cpu
{
CpuFlatten::CpuFlatten()
    : _reshape(nullptr)
{
}

CpuFlatten::~CpuFlatten() = default;

void CpuFlatten::configure(const ITensorInfo *src, ITensorInfo *dst)
{
    ARM_COMPUTE_LOG_PARAMS(src, dst);
    _reshape = std::make_unique<CpuReshape>();
    _reshape->configure(src, dst);
}

Status CpuFlatten::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    return CpuReshape::validate(src, dst);
}

void CpuFlatten::run(ITensorPack &tensors)
{
    _reshape->run(tensors);
}
} // namespace cpu
} // namespace arm_compute
