/*
 * Copyright (c) 2021 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEConv3D.h"

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "src/common/utils/Log.h"
#include "src/cpu/operators/CpuDirectConv3d.h"

namespace arm_compute
{
using namespace arm_compute::experimental;

struct NEConv3D::Impl
{
    std::unique_ptr<cpu::ICpuOperator> op{ nullptr };
    ITensorPack                        run_pack{};
};

NEConv3D::NEConv3D()
    : _impl(std::make_unique<Impl>())
{
}

NEConv3D::~NEConv3D() = default;

void NEConv3D::configure(ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const Conv3dInfo &conv_info)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_ERROR_THROW_ON(cpu::CpuDirectConv3d::validate(input->info(), weights->info(), ((biases != nullptr) ? biases->info() : nullptr), output->info(), conv_info));
    ARM_COMPUTE_LOG_PARAMS(input, weights, biases, output, conv_info);

    auto f = std::make_unique<cpu::CpuDirectConv3d>();
    f->configure(input->info(), weights->info(), ((biases != nullptr) ? biases->info() : nullptr), output->info(), conv_info);
    _impl->op = std::move(f);

    if(_impl->op != nullptr)
    {
        _impl->run_pack = { { ACL_SRC_0, input }, { ACL_SRC_1, weights }, { ACL_SRC_2, biases }, { ACL_DST, output } };
    }
}

Status NEConv3D::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const Conv3dInfo &conv_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(cpu::CpuDirectConv3d::validate(input, weights, biases, output, conv_info));

    return Status{};
}

void NEConv3D::run()
{
    if(_impl->op != nullptr)
    {
        _impl->op->run(_impl->run_pack);
    }
}
} // namespace arm_compute
