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

#include "arm_compute/runtime/experimental/operators/CpuGemm.h"

#include "src/cpu/operators/CpuGemm.h"

namespace arm_compute
{

namespace experimental
{
namespace ops
{

struct CpuGemm::Impl
{
    std::unique_ptr<arm_compute::cpu::CpuGemm> cpu_gemm{nullptr};
};

CpuGemm::CpuGemm() : _impl(std::make_unique<Impl>())
{
    _impl->cpu_gemm = std::make_unique<cpu::CpuGemm>();
}

CpuGemm::~CpuGemm() = default;

void CpuGemm::configure(const ITensorInfo *a,
                        const ITensorInfo *b,
                        const ITensorInfo *c,
                        ITensorInfo       *d,
                        float              alpha,
                        float              beta,
                        const GEMMInfo    &gemm_info)
{
    _impl->cpu_gemm->configure(a, b, c, d, alpha, beta, gemm_info);
}

Status CpuGemm::validate(const ITensorInfo *a,
                         const ITensorInfo *b,
                         const ITensorInfo *c,
                         const ITensorInfo *d,
                         float              alpha,
                         float              beta,
                         const GEMMInfo    &gemm_info)
{
    return cpu::CpuGemm::validate(a, b, c, d, alpha, beta, gemm_info);
}

Status CpuGemm::has_opt_impl(arm_compute::WeightFormat &weight_format,
                             const ITensorInfo         *a,
                             const ITensorInfo         *b,
                             const ITensorInfo         *c,
                             const ITensorInfo         *d,
                             const GEMMInfo            &gemm_info)
{
    return cpu::CpuGemm::has_opt_impl(weight_format, a, b, c, d, gemm_info);
}

void CpuGemm::run(ITensorPack &tensors)
{
    _impl->cpu_gemm->run(tensors);
}
void CpuGemm::prepare(ITensorPack &constants)
{
    _impl->cpu_gemm->prepare(constants);
}
experimental::MemoryRequirements CpuGemm::workspace() const
{
    return _impl->cpu_gemm->workspace();
}

} // namespace ops
} // namespace experimental
} // namespace arm_compute
