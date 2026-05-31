/*
 * Copyright (c) 2024, 2025-2026 Arm Limited.
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
#ifdef __aarch64__

#include "arm_gemm/arm_gemm.hpp"
#include "arm_gemm/gemm_common.hpp"
#include "gemm_hybrid_indirect.hpp"
#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"

#include "kernels/a64_hybrid_u8s8qa_dot_4x16.hpp"
#include "kernels/a64_hybrid_u8s8qa_mmla_4x16.hpp"
#include "kernels/a64_hybrid_u8s8s32_dot_6x16.hpp"
#include "kernels/a64_hybrid_u8s8s32_mmla_6x16.hpp"
#include "kernels/a64_interleaved_u8s8s32_mmla_8x12.hpp"
#ifdef ARM_COMPUTE_ENABLE_SVE
#include "kernels/sve_hybrid_u8s8qa_dot_4x4VL.hpp"
#include "kernels/sve_hybrid_u8s8qa_mmla_4x4VL.hpp"
#include "kernels/sve_hybrid_u8s8s32_mmla_6x4VL.hpp"
#include "kernels/sve_interleaved_u8s8s32_mmla_8x3VL.hpp"
#endif // ARM_COMPUTE_ENABLE_SVE

namespace arm_gemm {

static const GemmImplementation<uint8_t, int8_t, uint8_t, Requantize32> gemm_q8_mixed_methods[] =
{
#ifdef ARM_COMPUTE_ENABLE_SVE
GemmImplementation<uint8_t, int8_t, uint8_t, Requantize32>::with_estimate(
    "sve_hybrid_u8s8qa_mmla_4x4VL",
    [](const GemmArgs &args, const Requantize32 &qp) { return quant_hybrid_asymmetric(qp) && args._ci->has_sve2() && args._ci->has_svei8mm(); },
    [](const GemmArgs &args, const Requantize32 &) { return GemmHybridIndirect<cls_sve_hybrid_u8s8qa_mmla_4x4VL, uint8_t, int8_t, uint8_t, Requantize32>::estimate_cycles<int8_t>(args); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridIndirect<cls_sve_hybrid_u8s8qa_mmla_4x4VL, uint8_t, int8_t, uint8_t, Requantize32>(args, qp); }
),
GemmImplementation<uint8_t, int8_t, uint8_t, Requantize32>::with_estimate(
    "sve_interleaved_u8s8s32_mmla_8x3VL",
    [](const GemmArgs &args, const Requantize32 &) { return args._ci->has_svei8mm() && (args._Ksize>8); },
    [](const GemmArgs &args, const Requantize32 &) { return GemmInterleavedQuantized<cls_sve_interleaved_u8s8s32_mmla_8x3VL, uint8_t, int8_t, uint8_t>::estimate_cycles<uint8_t>(args); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmInterleavedQuantized<cls_sve_interleaved_u8s8s32_mmla_8x3VL, uint8_t, int8_t, uint8_t>(args, qp); }
),
GemmImplementation<uint8_t, int8_t, uint8_t, Requantize32>::with_estimate(
    "sve_hybrid_u8s8s32_mmla_6x4VL",
    [](const GemmArgs &args, const Requantize32 &) { return args._ci->has_svei8mm(); },
    [](const GemmArgs &args, const Requantize32 &) { return GemmHybridIndirect<cls_sve_hybrid_u8s8s32_mmla_6x4VL, uint8_t, int8_t, uint8_t, Requantize32, true>::estimate_cycles<uint8_t>(args); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridIndirect<cls_sve_hybrid_u8s8s32_mmla_6x4VL, uint8_t, int8_t, uint8_t, Requantize32, true>(args, qp); }
),
GemmImplementation<uint8_t, int8_t, uint8_t, Requantize32>::with_estimate(
    "sve_hybrid_u8s8qa_dot_4x4VL",
    [](const GemmArgs &args, const Requantize32 &qp) { return args._ci->has_sve2() && args._ci->has_svei8mm() && quant_hybrid_asymmetric(qp); },
    [](const GemmArgs &args, const Requantize32 &) { return GemmHybridIndirect<cls_sve_hybrid_u8s8qa_dot_4x4VL, uint8_t, int8_t, uint8_t, Requantize32>::estimate_cycles<int8_t>(args); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridIndirect<cls_sve_hybrid_u8s8qa_dot_4x4VL, uint8_t, int8_t, uint8_t, Requantize32>(args, qp); }
),
#endif // ARM_COMPUTE_ENABLE_SVE
GemmImplementation<uint8_t, int8_t, uint8_t, Requantize32>::with_estimate(
    "a64_hybrid_u8s8qa_mmla_4x16",
    [](const GemmArgs &args, const Requantize32 &qp) { return args._ci->has_i8mm() && quant_hybrid_asymmetric(qp); },
    [](const GemmArgs &args, const Requantize32 &) { return GemmHybridIndirect<cls_a64_hybrid_u8s8qa_mmla_4x16, uint8_t, int8_t, uint8_t, Requantize32>::estimate_cycles<int8_t>(args); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridIndirect<cls_a64_hybrid_u8s8qa_mmla_4x16, uint8_t, int8_t, uint8_t, Requantize32>(args, qp); }
),
GemmImplementation<uint8_t, int8_t, uint8_t, Requantize32>::with_estimate(
    "a64_interleaved_u8s8s32_mmla_8x12",
    [](const GemmArgs &args, const Requantize32 &) { return args._ci->has_i8mm() && (args._Ksize>8); },
    [](const GemmArgs &args, const Requantize32 &) { return GemmInterleavedQuantized<cls_a64_interleaved_u8s8s32_mmla_8x12, uint8_t, int8_t, uint8_t>::estimate_cycles<uint8_t>(args); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmInterleavedQuantized<cls_a64_interleaved_u8s8s32_mmla_8x12, uint8_t, int8_t, uint8_t>(args, qp); }
),
GemmImplementation<uint8_t, int8_t, uint8_t, Requantize32>::with_estimate(
    "a64_hybrid_u8s8s32_mmla_6x16",
    [](const GemmArgs &args, const Requantize32 &) { return args._ci->has_i8mm(); },
    [](const GemmArgs &args, const Requantize32 &) { return GemmHybridIndirect<cls_a64_hybrid_u8s8s32_mmla_6x16, uint8_t, int8_t, uint8_t, Requantize32, true>::estimate_cycles<uint8_t>(args); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridIndirect<cls_a64_hybrid_u8s8s32_mmla_6x16, uint8_t, int8_t, uint8_t, Requantize32, true>(args, qp); }
),
GemmImplementation<uint8_t, int8_t, uint8_t, Requantize32>::with_estimate(
    "a64_hybrid_u8s8qa_dot_4x16",
    [](const GemmArgs &args, const Requantize32 &qp) { return args._ci->has_i8mm() && quant_hybrid_asymmetric(qp); },
    [](const GemmArgs &args, const Requantize32 &) { return GemmHybridIndirect<cls_a64_hybrid_u8s8qa_dot_4x16, uint8_t, int8_t, uint8_t, Requantize32>::estimate_cycles<int8_t>(args); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridIndirect<cls_a64_hybrid_u8s8qa_dot_4x16, uint8_t, int8_t, uint8_t, Requantize32>(args, qp); }
),
GemmImplementation<uint8_t, int8_t, uint8_t, Requantize32>::with_estimate(
    "a64_hybrid_u8s8s32_dot_6x16",
    [](const GemmArgs &args, const Requantize32 &) { return args._ci->has_i8mm(); },
    [](const GemmArgs &args, const Requantize32 &) { return GemmHybridIndirect<cls_a64_hybrid_u8s8s32_dot_6x16, uint8_t, int8_t, uint8_t, Requantize32, true>::estimate_cycles<uint8_t>(args); },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridIndirect<cls_a64_hybrid_u8s8s32_dot_6x16, uint8_t, int8_t, uint8_t, Requantize32, true>(args, qp); }
),
{
    "",
    nullptr,
    nullptr,
    nullptr
}
};

template<>
const GemmImplementation<uint8_t, int8_t, uint8_t, Requantize32> *gemm_implementation_list<uint8_t, int8_t, uint8_t, Requantize32>() {
    return gemm_q8_mixed_methods;
}

template UniqueGemmCommon<uint8_t, int8_t, uint8_t> gemm<uint8_t, int8_t, uint8_t, Requantize32>(const GemmArgs &, const Requantize32 &);
template bool has_opt_gemm<uint8_t, int8_t, uint8_t, Requantize32>(WeightFormat &, const GemmArgs &, const Requantize32 &);
template std::vector<KernelDescription> get_compatible_kernels<uint8_t, int8_t, uint8_t, Requantize32>(const GemmArgs &, const Requantize32 &);

} // namespace arm_gemm

#endif // __aarch64__

