/*
 * Copyright (c) 2025-2026 Arm Limited.
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
#if defined(__aarch64__) && (defined(FP16_KERNELS) || defined(ARM_COMPUTE_ENABLE_FP16))

#include "arm_gemm/arm_gemm.hpp"
#include "arm_gemm/gemm_common.hpp"
#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"

#include "kernels/a64_gemm_s16_8x12.hpp"
#include "kernels/a64_gemm_s8_8x12.hpp"
#include "kernels/a64_interleaved_s8s32_mmla_8x12.hpp"
#ifdef ARM_COMPUTE_ENABLE_SVE
#include "kernels/sve_interleaved_s8s32_dot_8x3VL.hpp"
#include "kernels/sve_interleaved_s8s32_mmla_8x3VL.hpp"
#endif // ARM_COMPUTE_ENABLE_SVE

namespace arm_gemm {

static const GemmImplementation<int8_t, int8_t, __fp16, DequantizeFloat> gemm_s8fp16_methods[] =
{
#ifdef ARM_COMPUTE_ENABLE_SVE
GemmImplementation<int8_t, int8_t, __fp16, DequantizeFloat>::with_estimate(
    "sve_interleaved_s8s32_mmla_8x3VL",
    [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_svei8mm(); },
    [](const GemmArgs &args, const DequantizeFloat &) { return GemmInterleavedDequantized<cls_sve_interleaved_s8s32_mmla_8x3VL, int8_t,int8_t, __fp16>::estimate_cycles<int8_t>(args); },
    [](const GemmArgs &args, const DequantizeFloat &qp) { return new GemmInterleavedDequantized<cls_sve_interleaved_s8s32_mmla_8x3VL, int8_t,int8_t, __fp16>(args, qp); }
),
GemmImplementation<int8_t, int8_t, __fp16, DequantizeFloat>::with_estimate(
    "sve_interleaved_s8s32_dot_8x3VL",
    [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_sve(); },
    [](const GemmArgs &args, const DequantizeFloat &) { return GemmInterleavedDequantized<cls_sve_interleaved_s8s32_dot_8x3VL, int8_t,int8_t, __fp16>::estimate_cycles<int8_t>(args); },
    [](const GemmArgs &args, const DequantizeFloat &qp) { return new GemmInterleavedDequantized<cls_sve_interleaved_s8s32_dot_8x3VL, int8_t, int8_t, __fp16>(args, qp); }
),
#endif // ARM_COMPUTE_ENABLE_SVE
GemmImplementation<int8_t, int8_t, __fp16, DequantizeFloat>::with_estimate(
    "a64_interleaved_s8s32_mmla_8x12",
    [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_i8mm(); },
    [](const GemmArgs &args, const DequantizeFloat &) { return GemmInterleavedDequantized<cls_a64_interleaved_s8s32_mmla_8x12, int8_t, int8_t, __fp16>::estimate_cycles<int8_t>(args); },
    [](const GemmArgs &args, const DequantizeFloat &qp) { return new GemmInterleavedDequantized<cls_a64_interleaved_s8s32_mmla_8x12, int8_t, int8_t, __fp16>(args, qp); }
),
{
    "a64_gemm_s16_8x12",
    nullptr,
    [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->get_cpu_model() == CPUModel::A53 && ((args._Msize > 28) || ((args._Msize % 8) > 4)); },
    [](const GemmArgs &args, const DequantizeFloat &qp) { return new GemmInterleavedDequantized<cls_a64_gemm_s16_8x12, int8_t, int8_t, __fp16>(args, qp); }
},
GemmImplementation<int8_t, int8_t, __fp16, DequantizeFloat>::with_estimate(
    "a64_gemm_s8_8x12",
    [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_dotprod(); },
    [](const GemmArgs &args, const DequantizeFloat &) { return GemmInterleavedDequantized<cls_a64_gemm_s8_8x12, int8_t, int8_t, __fp16>::estimate_cycles<int8_t>(args); },
    [](const GemmArgs &args, const DequantizeFloat &qp) { return new GemmInterleavedDequantized<cls_a64_gemm_s8_8x12, int8_t, int8_t,  __fp16>(args, qp); }
),
{
    "",
    nullptr,
    nullptr,
    nullptr
}
};

template<>
const GemmImplementation<int8_t, int8_t, __fp16, DequantizeFloat> *gemm_implementation_list<int8_t, int8_t, __fp16, DequantizeFloat>() {
    return gemm_s8fp16_methods;
}

template UniqueGemmCommon<int8_t, int8_t, __fp16> gemm<int8_t, int8_t, __fp16, DequantizeFloat>(const GemmArgs &, const DequantizeFloat &);
template bool has_opt_gemm<int8_t, int8_t, __fp16, DequantizeFloat>(WeightFormat &, const GemmArgs &, const DequantizeFloat &);
template std::vector<KernelDescription> get_compatible_kernels<int8_t, int8_t, __fp16, DequantizeFloat>(const GemmArgs &, const DequantizeFloat &);

} // namespace arm_gemm

#endif // defined(__aarch64__) && (defined(FP16_KERNELS) || defined(ARM_COMPUTE_ENABLE_FP16))

