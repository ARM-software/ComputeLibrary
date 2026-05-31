/*
 * Copyright (c) 2017-2020, 2022-2026 Arm Limited.
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
#include "arm_common/bfloat.hpp"
#include "arm_gemm/arm_gemm.hpp"
#include "arm_gemm/gemm_common.hpp"
#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"

#ifdef __aarch64__
#ifdef ARM_COMPUTE_ENABLE_BF16
#ifdef ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
#include "kernels/a64_ffinterleaved_bf16fp32_mmla_8x12.hpp"
#ifdef ARM_COMPUTE_ENABLE_SVE
#include "kernels/sve_ffinterleaved_bf16fp32_mmla_8x3VL.hpp"
#endif // ARM_COMPUTE_ENABLE_SVE
#endif // ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
#endif // ARM_COMPUTE_ENABLE_BF16
#endif // __aarch64__

namespace arm_gemm {

static const GemmImplementation<bfloat16, bfloat16, bfloat16> gemm_bf16bf16_methods[] =
{
#ifdef __aarch64__
#ifdef ARM_COMPUTE_ENABLE_BF16
#ifdef ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
GemmImplementation<bfloat16, bfloat16, bfloat16>::with_estimate(
    "a64_ffinterleaved_bf16fp32_mmla_8x12",
    KernelWeightFormat::VL256_BL64,
    [](const GemmArgs &args) { return args._ci->has_bf16(); },
    [](const GemmArgs &args) { return GemmInterleavedFixedFormat<cls_a64_ffinterleaved_bf16fp32_mmla_8x12, bfloat16, bfloat16, bfloat16>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmInterleavedFixedFormat<cls_a64_ffinterleaved_bf16fp32_mmla_8x12, bfloat16, bfloat16, bfloat16>(args); }
),
#ifdef ARM_COMPUTE_ENABLE_SVE
GemmImplementation<bfloat16, bfloat16, bfloat16>::with_estimate(
    "sve_ffinterleaved_bf16fp32_mmla_8x3VL",
    KernelWeightFormat::VL2VL_BL64,
    [](const GemmArgs &args) { return args._ci->has_svebf16(); },
    [](const GemmArgs &args) { return GemmInterleavedFixedFormat<cls_sve_ffinterleaved_bf16fp32_mmla_8x3VL, bfloat16, bfloat16, bfloat16>::estimate_cycles<bfloat16>(args); },
    [](const GemmArgs &args) { return new GemmInterleavedFixedFormat<cls_sve_ffinterleaved_bf16fp32_mmla_8x3VL, bfloat16, bfloat16, bfloat16>(args); }
),
#endif // ARM_COMPUTE_ENABLE_SVE
#endif // ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
#endif // ARM_COMPUTE_ENABLE_BF16
#endif // __aarch64__
{
    "",
    nullptr,
    nullptr,
    nullptr
}
};

template<>
const GemmImplementation<bfloat16, bfloat16, bfloat16> *gemm_implementation_list<bfloat16, bfloat16, bfloat16>() {
    return gemm_bf16bf16_methods;
}

template UniqueGemmCommon<bfloat16, bfloat16, bfloat16> gemm<bfloat16, bfloat16, bfloat16, Nothing>(const GemmArgs &, const Nothing &);
template bool has_opt_gemm<bfloat16, bfloat16, bfloat16, Nothing>(WeightFormat &, const GemmArgs &, const Nothing &);
template std::vector<KernelDescription> get_compatible_kernels<bfloat16, bfloat16, bfloat16, Nothing>(const GemmArgs &, const Nothing &);

} // namespace arm_gemm

