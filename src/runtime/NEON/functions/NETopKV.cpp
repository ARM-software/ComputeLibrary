/*
 * Copyright (c) 2026 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NETopKV.h"

#include "arm_compute/core/Validate.h"

#include "src/common/utils/Log.h"
#include "src/common/utils/profile/acl_profile.h"
#include "src/cpu/operators/CpuTopKV.h"

namespace arm_compute
{
struct NETopKV::Impl
{
    const ITensor                 *predictions{nullptr};
    const ITensor                 *targets{nullptr};
    ITensor                       *output{nullptr};
    unsigned int                   k{0};
    std::unique_ptr<cpu::CpuTopKV> op{nullptr};
};

NETopKV::NETopKV() : _impl(std::make_unique<Impl>())
{
}
NETopKV::NETopKV(NETopKV &&)            = default;
NETopKV &NETopKV::operator=(NETopKV &&) = default;
NETopKV::~NETopKV()                     = default;

void NETopKV::configure(const ITensor *predictions, const ITensor *targets, ITensor *output, const unsigned int k)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "NETopKV::configure");
    ARM_COMPUTE_LOG_PARAMS(predictions, targets, output, k);

    _impl->predictions = predictions;
    _impl->targets     = targets;
    _impl->output      = output;
    _impl->k           = k;

    ARM_COMPUTE_ERROR_ON_NULLPTR(_impl->predictions, _impl->targets, _impl->output);

    _impl->op = std::make_unique<cpu::CpuTopKV>();
    _impl->op->configure(_impl->predictions->info(), _impl->targets->info(), _impl->output->info(), k);
}

Status
NETopKV::validate(const ITensorInfo *predictions, const ITensorInfo *targets, ITensorInfo *output, const unsigned int k)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "NETopKV::validate");
    ARM_COMPUTE_RETURN_ERROR_ON_DYNAMIC_SHAPE(predictions, targets, output);
    return cpu::CpuTopKV::validate(predictions, targets, output, k);
}

void NETopKV::run()
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "NETopKV::run");
    ARM_COMPUTE_LOG_PARAMS(_impl->predictions, _impl->targets, _impl->output, _impl->k);

    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->predictions);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->targets);
    pack.add_tensor(TensorType::ACL_DST, _impl->output);
    _impl->op->run(pack);
}
} // namespace arm_compute
