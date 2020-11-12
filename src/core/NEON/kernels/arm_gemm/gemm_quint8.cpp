/*
 * Copyright (c) 2019-2020 Arm Limited.
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

#include "kernels/a64_gemm_u16_8x12.hpp"
#include "kernels/a64_gemm_u8_4x4.hpp"
#include "kernels/a64_gemm_u8_8x12.hpp"
#include "kernels/a64_hybrid_u8qa_dot_4x16.hpp"
#include "kernels/a64_hybrid_u8u32_dot_6x16.hpp"
#include "kernels/a64_interleaved_u8u32_mmla_8x12.hpp"
#include "kernels/a64_smallK_hybrid_u8u32_dot_6x4.hpp"
#include "kernels/a64_smallK_hybrid_u8u32_dot_8x4.hpp"

#include "kernels/sve_hybrid_u8u32_dot_6x4VL.hpp"
#include "kernels/sve_hybrid_u8qa_dot_4x4VL.hpp"
#include "kernels/sve_interleaved_u8u32_dot_8x3VL.hpp"
#include "kernels/sve_interleaved_u8u32_mmla_8x3VL.hpp"
#include "kernels/sve_smallK_hybrid_u8u32_dot_8x1VL.hpp"

#include "gemm_hybrid_indirect.hpp"
#include "gemm_hybrid_quantized.hpp"
#include "gemm_hybrid_quantized_inline.hpp"
#include "gemm_interleaved.hpp"
#include "quantize_wrapper.hpp"

namespace arm_gemm {

static const GemmImplementation<uint8_t, uint8_t, Requantize32> gemm_quint8_methods[] =
{
#ifdef __ARM_FEATURE_SVE
#ifdef MMLA_INT8
{
    GemmMethod::GEMM_INTERLEAVED,
    "sve_interleaved_u8u32_mmla_8x3VL",
    [](const GemmArgs &args, const Requantize32 &) { return (args._Ksize>8); },
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmInterleavedQuantized<cls_sve_interleaved_u8u32_mmla_8x3VL, uint8_t, uint8_t>(args, qp); }
},
#endif
{
    GemmMethod::GEMM_HYBRID_QUANTIZED,
    "sve_smallK_hybrid_u8u32_dot_8x1VL",
    [](const GemmArgs &args, const Requantize32 &) { return args._Ksize<=64 && !args._indirect_input; },
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridQuantized<cls_sve_smallK_hybrid_u8u32_dot_8x1VL, uint8_t, uint8_t>(args, qp); }
},
#ifdef SVE2 // Requantizing kernels include some SVE2 only instructions (SQRDMULH, SRSHL)
{
    GemmMethod::GEMM_HYBRID, 
    "sve_hybrid_u8qa_dot_4x4VL",
    [](const GemmArgs &args, const Requantize32 &qp) { return quant_hybrid_asymmetric(qp); },
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridIndirect<cls_sve_hybrid_u8qa_dot_4x4VL, uint8_t, uint8_t, Requantize32>(args, qp); }
},
#endif
{
    GemmMethod::GEMM_HYBRID, 
    "sve_hybrid_u8u32_dot_6x4VL",
    nullptr,
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridIndirect<cls_sve_hybrid_u8u32_dot_6x4VL, uint8_t, uint8_t, Requantize32, true>(args, qp); }
},
{
    GemmMethod::GEMM_INTERLEAVED,
    "sve_interleaved_u8u32_dot_8x3VL",
    [](const GemmArgs &args, const Requantize32 &) { return (args._Ksize>4); },
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmInterleavedQuantized<cls_sve_interleaved_u8u32_dot_8x3VL, uint8_t, uint8_t>(args, qp); }
},
#endif
#ifdef MMLA_INT8
{
    GemmMethod::GEMM_INTERLEAVED,
    "a64_interleaved_u8u32_mmla_8x12",
    [](const GemmArgs &args, const Requantize32 &) { return (args._Ksize>8); },
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmInterleavedQuantized<cls_a64_interleaved_u8u32_mmla_8x12, uint8_t, uint8_t>(args, qp); }
},
#endif
{
    GemmMethod::GEMM_HYBRID_QUANTIZED,
    "a64_smallK_hybrid_u8u32_dot_8x4",
    [](const GemmArgs &args, const Requantize32 &) { return args._ci->has_dotprod() && (args._Nsize % 4 == 0) && (args._Ksize<=32) && !args._indirect_input; },
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridQuantized<cls_a64_smallK_hybrid_u8u32_dot_8x4, uint8_t, uint8_t>(args, qp); }
},
{
    GemmMethod::GEMM_HYBRID_QUANTIZED,
    "a64_smallK_hybrid_u8u32_dot_6x4",
    [](const GemmArgs &args, const Requantize32 &) { return args._ci->has_dotprod() && (args._Nsize % 4 == 0) && (args._Ksize>32) && (args._Ksize<=64) && !args._indirect_input; },
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridQuantized<cls_a64_smallK_hybrid_u8u32_dot_6x4, uint8_t, uint8_t>(args, qp); }
},
{
    GemmMethod::GEMM_INTERLEAVED,
    "a64_gemm_u16_8x12",
    nullptr,
    [](const GemmArgs &args, const Requantize32 &) { return args._ci->get_cpu_model() == CPUModel::A53; },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmInterleavedQuantized<cls_a64_gemm_u16_8x12, uint8_t, uint8_t>(args, qp); },
},
{
    GemmMethod::GEMM_HYBRID,
    "a64_hybrid_u8qa_dot_4x16",
    [](const GemmArgs &args, const Requantize32 &qp) { return args._ci->has_dotprod() && quant_hybrid_asymmetric(qp); },
    [](const GemmArgs &args, const Requantize32 &) { return args._Nsize<=256 && args._Ksize>128; },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridIndirect<cls_a64_hybrid_u8qa_dot_4x16, uint8_t, uint8_t, Requantize32>(args, qp); }
},
{
    GemmMethod::GEMM_HYBRID,
    "a64_hybrid_u8u32_dot_6x16",
    [](const GemmArgs &args, const Requantize32 &) { return args._ci->has_dotprod(); },
    [](const GemmArgs &args, const Requantize32 &) { return args._Nsize<=256 && args._Ksize>128; },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridIndirect<cls_a64_hybrid_u8u32_dot_6x16, uint8_t, uint8_t, Requantize32, true>(args, qp); }
},
{
    GemmMethod::GEMM_INTERLEAVED,
    "a64_gemm_u8_8x12",
    [](const GemmArgs &args, const Requantize32 &) { return args._ci->has_dotprod(); },
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmInterleavedQuantized<cls_a64_gemm_u8_8x12, uint8_t, uint8_t>(args, qp); }
},
{
    GemmMethod::GEMM_INTERLEAVED,
    "a64_gemm_u8_4x4",
    nullptr,
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmInterleavedQuantized<cls_a64_gemm_u8_4x4, uint8_t, uint8_t>(args, qp); }
},
{
    GemmMethod::QUANTIZE_WRAPPER,
    "quantized_wrapper",
    [](const GemmArgs &args, const Requantize32 &) { return !args._indirect_input; },
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new QuantizeWrapper<uint8_t, uint8_t, uint32_t>(args, qp); }
},
{
    GemmMethod::DEFAULT,
    "",
    nullptr,
    nullptr,
    nullptr
}
};

template<>
const GemmImplementation<uint8_t, uint8_t, Requantize32> *gemm_implementation_list<uint8_t, uint8_t, Requantize32>() {
    return gemm_quint8_methods;
}

template UniqueGemmCommon<uint8_t, uint8_t> gemm<uint8_t, uint8_t, Requantize32>(const GemmArgs &args, const Requantize32 &os);
template KernelDescription get_gemm_method<uint8_t, uint8_t, Requantize32>(const GemmArgs &args, const Requantize32 &os);
template std::vector<KernelDescription> get_compatible_kernels<uint8_t, uint8_t, Requantize32>(const GemmArgs &args, const Requantize32 &os);

} // namespace arm_gemm

#endif // __aarch64__
