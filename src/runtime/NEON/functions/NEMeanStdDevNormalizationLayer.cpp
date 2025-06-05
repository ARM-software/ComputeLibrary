/*
 * Copyright (c) 2019-2021, 2024-2025 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEMeanStdDevNormalizationLayer.h"

#include "arm_compute/core/Validate.h"

#include "src/common/utils/Log.h"
#include "src/common/utils/profile/acl_profile.h"
#include "src/cpu/operators/CpuMeanStdDevNormalization.h"

namespace arm_compute
{
struct arm_compute::NEMeanStdDevNormalizationLayer::Impl
{
    ITensor                                         *input{nullptr};
    ITensor                                         *output{nullptr};
    std::unique_ptr<cpu::CpuMeanStdDevNormalization> op{nullptr};
};

NEMeanStdDevNormalizationLayer::NEMeanStdDevNormalizationLayer() : _impl(std::make_unique<Impl>())
{
}

NEMeanStdDevNormalizationLayer::~NEMeanStdDevNormalizationLayer() = default;

void NEMeanStdDevNormalizationLayer::configure(ITensor *input, ITensor *output, float epsilon)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU,
                            "NEMeanStdDevNormalizationLayer::configure");
    _impl->input  = input;
    _impl->output = (output == nullptr) ? input : output;
    _impl->op     = std::make_unique<cpu::CpuMeanStdDevNormalization>();
    _impl->op->configure(_impl->input->info(), _impl->output->info(), epsilon);
}

Status NEMeanStdDevNormalizationLayer::validate(const ITensorInfo *input, const ITensorInfo *output, float epsilon)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU,
                            "NEMeanStdDevNormalizationLayer::validate");
    return cpu::CpuMeanStdDevNormalization::validate(input, output, epsilon);
}

void NEMeanStdDevNormalizationLayer::run()
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "NEMeanStdDevNormalizationLayer::run");
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->input);
    pack.add_tensor(TensorType::ACL_DST, _impl->output);
    _impl->op->run(pack);
}
} // namespace arm_compute
