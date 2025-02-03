/*
 * Copyright (c) 2024-2025 Arm Limited.
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

#include "src/cpu/kernels/CpuDynamicGemmKernelHeuristics.h"

#include "arm_compute/core/utils/DataLayoutUtils.h"

#include "src/core/common/Registrars.h"
#include "src/cpu/kernels/CpuDynamicGemmKernel.h"

#if defined(__aarch64__)
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.h"
#endif // __aarch64__

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace heuristics
{
namespace
{

#if defined(__aarch64__) && defined(ENABLE_FP32_KERNELS)
void neon_fp32_dynamic_gemm(const ITensor *a,
                            const ITensor *b,
                            const ITensor *c,
                            ITensor       *d,
                            ITensor       *pack_b_and_c_output,
                            const Window  &window)
{
    ARM_COMPUTE_UNUSED(window);

    if (pack_b_and_c_output != nullptr)
    {
        const size_t      num_groups  = 1;
        const size_t      n           = b->info()->tensor_shape().x();
        const size_t      k           = b->info()->tensor_shape().y();
        const size_t      nr          = kai_get_nr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla();
        const size_t      kr          = kai_get_kr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla();
        const size_t      sr          = kai_get_sr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla();
        const size_t      rhs_stride  = b->info()->strides_in_bytes().y();
        const void *const rhs         = b->buffer() + b->info()->offset_first_element_in_bytes();
        const void *const bias        = c->buffer() + c->info()->offset_first_element_in_bytes();
        const void *const scale       = nullptr;
        void *const       rhs_packed  = pack_b_and_c_output->buffer();
        const size_t      extra_bytes = 0;
        const void *const params      = nullptr;
        kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(num_groups, n, k, nr, kr, sr, rhs_stride, rhs, bias, scale,
                                                         rhs_packed, extra_bytes, params);
        b = pack_b_and_c_output;
    }

    const size_t      m              = d->info()->tensor_shape().y();
    const size_t      n              = d->info()->tensor_shape().x();
    const size_t      k              = a->info()->tensor_shape().x();
    const void *const lhs            = a->buffer() + a->info()->offset_first_element_in_bytes();
    const size_t      lhs_stride     = a->info()->strides_in_bytes().y();
    const void *const rhs_packed     = b->buffer();
    void *const       dst            = d->buffer() + d->info()->offset_first_element_in_bytes();
    const size_t      dst_stride_row = d->info()->strides_in_bytes().y();
    const size_t      dst_stride_col = d->info()->strides_in_bytes().x();
    const float       clamp_min      = -std::numeric_limits<float>::max();
    const float       clamp_max      = std::numeric_limits<float>::max();
    kai_run_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla(m, n, k, lhs, lhs_stride, rhs_packed, dst,
                                                               dst_stride_row, dst_stride_col, clamp_min, clamp_max);
}
#endif // __aarch64__ && ENABLE_FP32_KERNELS

} // namespace

const CpuDynamicGemmKernelHeuristics::KernelList CpuDynamicGemmKernelHeuristics::fp32_kernels
{
#if defined(__aarch64__)
    {"neon_fp32_dynamic_gemm",
     [](const DataTypeISASelectorData &data)
     {
         ARM_COMPUTE_UNUSED(data);
         return true;
     },
     REGISTER_FP32_NEON(neon_fp32_dynamic_gemm)},
#endif /* __aarch64__ */
};

const CpuDynamicGemmKernelHeuristics::KernelMap CpuDynamicGemmKernelHeuristics::kernels{
    {DataType::F32, fp32_kernels},
};

void CpuDynamicGemmKernelHeuristics::choose_kernel(const DataTypeISASelectorData &selector)
{
    const auto &klist = kernels.find(selector.dt);
    ARM_COMPUTE_ERROR_ON(klist == kernels.end());

    for (const auto &uk : klist->second)
    {
        if (uk.is_selected(selector))
        {
            _kernel = &uk;
            return;
        }
    }
}

CpuDynamicGemmKernelHeuristics::CpuDynamicGemmKernelHeuristics(const ITensorInfo *a,
                                                               const ITensorInfo *b,
                                                               const ITensorInfo *c,
                                                               ITensorInfo       *d,
                                                               float              alpha,
                                                               float              beta,
                                                               const GEMMInfo    &gemm_info)
{
    ARM_COMPUTE_UNUSED(b);
    ARM_COMPUTE_UNUSED(c);
    ARM_COMPUTE_UNUSED(d);
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_UNUSED(beta);
    ARM_COMPUTE_UNUSED(gemm_info);

    const DataTypeISASelectorData selector{a->data_type(), CPUInfo::get().get_isa()};
    choose_kernel(selector);
}

/** Return minimum workload size
 *
 * @return Minimum workload size for requested configuration.
 */
size_t CpuDynamicGemmKernelHeuristics::mws() const
{
    return _mws;
}

/** Return kernel's execution window
 *
 * @return The execution window
 */
const Window &CpuDynamicGemmKernelHeuristics::window() const
{
    return _window;
}

/** Return the kernel to run
 *
 * @return The function pointer to the chosen kernel
 */
CpuDynamicGemmKernelHeuristics::KernelPtr CpuDynamicGemmKernelHeuristics::kernel() const
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(_kernel);
    return _kernel->ukernel;
}

const char *CpuDynamicGemmKernelHeuristics::name() const
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(_kernel);
    return _kernel->name;
}

/** Return the scheduling hint e.g. dimension(s) to split
 *
 * @return an instance of @ref IScheduler::Hints to describe the scheduling hints
 */
const IScheduler::Hints &CpuDynamicGemmKernelHeuristics::scheduler_hint() const
{
    return _hint;
}
} // namespace heuristics
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
