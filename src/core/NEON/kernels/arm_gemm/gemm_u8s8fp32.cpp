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
#ifdef __aarch64__

#include "arm_gemm.hpp"

#include "kernels/a64_hybrid_u8s8qa_dot_4x16.hpp"
#include "kernels/a64_hybrid_u8s8qa_mmla_4x16.hpp"
#include "kernels/a64_hybrid_u8s8s32_dot_6x16.hpp"
#include "kernels/a64_hybrid_u8s8s32_mmla_6x16.hpp"
#include "kernels/a64_interleaved_u8s8s32_mmla_8x12.hpp"

#ifdef ARM_COMPUTE_ENABLE_SVE
#include "kernels/sve_hybrid_u8s8qa_dot_4x4VL.hpp"
#include "kernels/sve_interleaved_u8s8s32_mmla_8x3VL.hpp"
#include "kernels/sve_hybrid_u8s8s32_mmla_6x4VL.hpp"
#include "kernels/sve_hybrid_u8s8qa_mmla_4x4VL.hpp"
#endif // ARM_COMPUTE_ENABLE_SVE

#include "gemm_hybrid_indirect.hpp"
#include "gemm_hybrid_quantized.hpp"
#include "gemm_interleaved.hpp"
#include "gemv_pretransposed.hpp"
#include "quantize_wrapper.hpp"
#include "utils.hpp"

namespace arm_gemm {

static const GemmImplementation<uint8_t, int8_t, float, DequantizeFloat> gemm_u8s8fp32_methods[] =
{
#ifdef ARM_COMPUTE_ENABLE_SVE
GemmImplementation<uint8_t, int8_t, float, DequantizeFloat>::with_estimate(
    GemmMethod::GEMM_INTERLEAVED,
    "sve_interleaved_u8s8s32_mmla_8x3VL",
    [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_svei8mm(); },
    [](const GemmArgs &args, const DequantizeFloat &) { return GemmInterleavedDequantized<cls_sve_interleaved_u8s8s32_mmla_8x3VL, uint8_t, int8_t, float>::estimate_cycles<uint8_t>(args); },
    [](const GemmArgs &args, const DequantizeFloat &qp) { return new GemmInterleavedDequantized<cls_sve_interleaved_u8s8s32_mmla_8x3VL, uint8_t, int8_t, float>(args, qp); }
),
#endif // ARM_COMPUTE_ENABLE_SVE
GemmImplementation<uint8_t, int8_t, float, DequantizeFloat>::with_estimate(
    GemmMethod::GEMM_INTERLEAVED,
    "a64_interleaved_u8s8s32_mmla_8x12",
    [](const GemmArgs &args, const DequantizeFloat &) { return args._ci->has_i8mm(); },
    [](const GemmArgs &args, const DequantizeFloat &) { return GemmInterleavedDequantized<cls_a64_interleaved_u8s8s32_mmla_8x12, uint8_t, int8_t, float>::estimate_cycles<uint8_t>(args); },
    [](const GemmArgs &args, const DequantizeFloat &qp) { return new GemmInterleavedDequantized<cls_a64_interleaved_u8s8s32_mmla_8x12, uint8_t, int8_t, float>(args, qp); }
),
{
    GemmMethod::DEFAULT,
    "",
    nullptr,
    nullptr,
    nullptr
}
};

template<>
const GemmImplementation<uint8_t, int8_t, float, DequantizeFloat> *gemm_implementation_list<uint8_t, int8_t, float, DequantizeFloat>() {
    return gemm_u8s8fp32_methods;
}

template UniqueGemmCommon<uint8_t, int8_t, float> gemm<uint8_t, int8_t, float, DequantizeFloat>(const GemmArgs &args, const DequantizeFloat &os);
template bool has_opt_gemm<uint8_t, int8_t, float, DequantizeFloat>(WeightFormat &weight_format, const GemmArgs &args, const DequantizeFloat &os);
template KernelDescription get_gemm_method<uint8_t, int8_t, float, DequantizeFloat>(const GemmArgs &args, const DequantizeFloat &os);
template std::vector<KernelDescription> get_compatible_kernels<uint8_t, int8_t, float, DequantizeFloat>(const GemmArgs &args, const DequantizeFloat &os);

} // namespace arm_gemm

#endif // __aarch64__
