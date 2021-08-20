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
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpOutputStage.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Validate.h"
#include "src/cpu/operators/CpuGemmLowpOutputStage.h"

namespace arm_compute
{
struct NEGEMMLowpOutputStage::Impl
{
    const ITensor                               *src{ nullptr };
    const ITensor                               *bias{ nullptr };
    ITensor                                     *dst{ nullptr };
    ITensorPack                                  run_pack{};
    std::unique_ptr<cpu::CpuGemmLowpOutputStage> op{ nullptr };
};

NEGEMMLowpOutputStage::NEGEMMLowpOutputStage()
    : _impl(std::make_unique<Impl>())
{
}
NEGEMMLowpOutputStage::~NEGEMMLowpOutputStage() = default;

void NEGEMMLowpOutputStage::configure(const ITensor *input, const ITensor *bias, ITensor *output, const GEMMLowpOutputStageInfo &info)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(NEGEMMLowpOutputStage::validate(input->info(), bias != nullptr ? bias->info() : nullptr, output->info(), info));
    _impl->src  = input;
    _impl->bias = bias;
    _impl->dst  = output;
    _impl->op   = std::make_unique<cpu::CpuGemmLowpOutputStage>();
    _impl->op->configure(input->info(), (bias == nullptr) ? nullptr : bias->info(), output->info(), info);

    _impl->run_pack =
    {
        { TensorType::ACL_SRC, _impl->src },
        { TensorType::ACL_BIAS, _impl->bias },
        { TensorType::ACL_DST, _impl->dst }
    };
}

Status NEGEMMLowpOutputStage::validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, const GEMMLowpOutputStageInfo &info)
{
    return cpu::CpuGemmLowpOutputStage::validate(input, bias, output, info);
}

void NEGEMMLowpOutputStage::run()
{
    _impl->op->run(_impl->run_pack);
}
} // namespace arm_compute
