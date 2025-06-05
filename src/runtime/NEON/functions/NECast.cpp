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
#include "arm_compute/runtime/NEON/functions/NECast.h"

#include "arm_compute/core/Validate.h"

#include "src/common/utils/Log.h"
#include "src/common/utils/profile/acl_profile.h"
#include "src/cpu/operators/CpuCast.h"

namespace arm_compute
{
struct NECast::Impl
{
    const ITensor                *src{nullptr};
    ITensor                      *dst{nullptr};
    std::unique_ptr<cpu::CpuCast> op{nullptr};
};

NECast::NECast() : _impl(std::make_unique<Impl>())
{
}
NECast::NECast(NECast &&)            = default;
NECast &NECast::operator=(NECast &&) = default;
NECast::~NECast()                    = default;

void NECast::configure(ITensor *input, ITensor *output, ConvertPolicy policy)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "NECast::configure");
    _impl->src = input;
    _impl->dst = output;

    ARM_COMPUTE_ERROR_ON_NULLPTR(_impl->src, _impl->dst);
    ARM_COMPUTE_LOG_PARAMS(input, output, policy);
    _impl->op = std::make_unique<cpu::CpuCast>();
    _impl->op->configure(_impl->src->info(), _impl->dst->info(), policy);
}

Status NECast::validate(const ITensorInfo *input, const ITensorInfo *output, ConvertPolicy policy)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "NECast::validate");
    ARM_COMPUTE_RETURN_ERROR_ON_DYNAMIC_SHAPE(input, output);
    return cpu::CpuCast::validate(input, output, policy);
}

void NECast::run()
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "NECast::run");
    ITensorPack pack = {{ACL_SRC, _impl->src}, {ACL_DST, _impl->dst}};
    _impl->op->run(pack);
}
} // namespace arm_compute
