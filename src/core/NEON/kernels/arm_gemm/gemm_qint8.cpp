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

#include "kernels/a64_gemm_s16_8x12.hpp"
#include "kernels/a64_gemm_s8_4x4.hpp"
#include "kernels/a64_gemm_s8_8x12.hpp"
#include "kernels/a64_hybrid_s8qa_dot_4x16.hpp"
#include "kernels/a64_hybrid_s8qs_dot_6x16.hpp"
#include "kernels/a64_hybrid_s8s32_dot_6x16.hpp"
#include "kernels/a64_interleaved_s8s32_mmla_8x12.hpp"
#include "kernels/a64_smallK_hybrid_s8s32_dot_6x4.hpp"
#include "kernels/a64_smallK_hybrid_s8s32_dot_8x4.hpp"

#include "kernels/sve_hybrid_s8s32_dot_6x4VL.hpp"
#include "kernels/sve_hybrid_s8qa_dot_4x4VL.hpp"
#include "kernels/sve_hybrid_s8qs_dot_6x4VL.hpp"
#include "kernels/sve_interleaved_s8s32_dot_8x3VL.hpp"
#include "kernels/sve_interleaved_s8s32_mmla_8x3VL.hpp"
#include "kernels/sve_smallK_hybrid_s8s32_dot_8x1VL.hpp"

#include "gemm_hybrid_indirect.hpp"
#include "gemm_hybrid_quantized.hpp"
#include "gemm_hybrid_quantized_inline.hpp"
#include "gemm_interleaved.hpp"
#include "quantize_wrapper.hpp"
#include "utils.hpp"

namespace arm_gemm {

static const GemmImplementation<int8_t, int8_t, Requantize32> gemm_qint8_methods[] =
{
#ifdef __ARM_FEATURE_SVE
#ifdef MMLA_INT8
{
    GemmMethod::GEMM_INTERLEAVED,
    "sve_interleaved_s8s32_mmla_8x3VL",
    [](const GemmArgs &args, const Requantize32 &) { return (args._Ksize>8); },
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmInterleavedQuantized<cls_sve_interleaved_s8s32_mmla_8x3VL, int8_t, int8_t>(args, qp); }
},
#endif
{
    GemmMethod::GEMM_HYBRID_QUANTIZED,
    "sve_smallK_hybrid_s8s32_dot_8x1VL",
    [](const GemmArgs &args, const Requantize32 &) { return args._Ksize<=64 && !args._indirect_input; },
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridQuantized<cls_sve_smallK_hybrid_s8s32_dot_8x1VL, int8_t, int8_t>(args, qp); }
},
#ifdef SVE2
{
    GemmMethod::GEMM_HYBRID,
    "sve_hybrid_s8qs_dot_6x4VL",
    [](const GemmArgs &args, const Requantize32 &qp) { return quant_hybrid_symmetric(qp); },
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridIndirect<cls_sve_hybrid_s8qs_dot_6x4VL, int8_t, int8_t, Requantize32>(args, qp); }
},
{
    GemmMethod::GEMM_HYBRID,
    "sve_hybrid_s8qa_dot_4x4VL",
    [](const GemmArgs &args, const Requantize32 &qp) { return quant_hybrid_asymmetric(qp); },
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridIndirect<cls_sve_hybrid_s8qa_dot_4x4VL, int8_t, int8_t, Requantize32>(args, qp); }
},
#endif
{
    GemmMethod::GEMM_HYBRID,
    "sve_hybrid_s8s32_dot_6x4VL",
    nullptr,
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridIndirect<cls_sve_hybrid_s8s32_dot_6x4VL, int8_t, int8_t, Requantize32, true>(args, qp); }
},
{
    GemmMethod::GEMM_INTERLEAVED,
    "sve_interleaved_s8s32_dot_8x3VL",
    [](const GemmArgs &args, const Requantize32 &) { return (args._Ksize>4); },
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmInterleavedQuantized<cls_sve_interleaved_s8s32_dot_8x3VL, int8_t, int8_t>(args, qp); }
},
#endif // SVE
#ifdef MMLA_INT8
{
    GemmMethod::GEMM_INTERLEAVED,
    "a64_interleaved_s8s32_mmla_8x12",
    [](const GemmArgs &args, const Requantize32 &) { return (args._Ksize>8); },
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmInterleavedQuantized<cls_a64_interleaved_s8s32_mmla_8x12, int8_t, int8_t>(args, qp); }
},
#endif
{
    GemmMethod::GEMM_HYBRID_QUANTIZED,
    "a64_smallK_hybrid_s8s32_dot_8x4",
    [](const GemmArgs &args, const Requantize32 &) { return args._ci->has_dotprod() && (args._Nsize % 4 == 0) && (args._Ksize<=32) && !args._indirect_input; },
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridQuantized<cls_a64_smallK_hybrid_s8s32_dot_8x4, int8_t, int8_t>(args, qp); }
},
{
    GemmMethod::GEMM_HYBRID_QUANTIZED,
    "a64_smallK_hybrid_s8s32_dot_6x4",
    [](const GemmArgs &args, const Requantize32 &) { return args._ci->has_dotprod() && (args._Nsize % 4 == 0) && (args._Ksize>32) && (args._Ksize<=64) && !args._indirect_input; },
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridQuantized<cls_a64_smallK_hybrid_s8s32_dot_6x4, int8_t, int8_t>(args, qp); }
},
{
    GemmMethod::GEMM_INTERLEAVED,
    "a64_gemm_s16_8x12",
    nullptr,
    [](const GemmArgs &args, const Requantize32 &) { return args._ci->get_cpu_model() == CPUModel::A53; },
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmInterleavedQuantized<cls_a64_gemm_s16_8x12, int8_t, int8_t>(args, qp); }
},
{
    GemmMethod::GEMM_HYBRID,
    "a64_hybrid_s8qs_dot_6x16",
    [](const GemmArgs &args, const Requantize32 &qp) { return args._ci->has_dotprod() && quant_hybrid_symmetric(qp); },
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridIndirect<cls_a64_hybrid_s8qs_dot_6x16, int8_t, int8_t, Requantize32>(args, qp); }
},
{
    GemmMethod::GEMM_HYBRID,
    "a64_hybrid_s8qa_dot_4x16",
    [](const GemmArgs &args, const Requantize32 &qp) { return args._ci->has_dotprod() && quant_hybrid_asymmetric(qp); },
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridIndirect<cls_a64_hybrid_s8qa_dot_4x16, int8_t, int8_t, Requantize32>(args, qp); }
},
{
    GemmMethod::GEMM_HYBRID,
    "a64_hybrid_s8s32_dot_6x16",
    [](const GemmArgs &args, const Requantize32 &) { return args._ci->has_dotprod(); },
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmHybridIndirect<cls_a64_hybrid_s8s32_dot_6x16, int8_t, int8_t, Requantize32, true>(args, qp); }
},
{
    GemmMethod::GEMM_INTERLEAVED,
    "a64_gemm_s8_8x12",
    [](const GemmArgs &args, const Requantize32 &) { return args._ci->has_dotprod(); },
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmInterleavedQuantized<cls_a64_gemm_s8_8x12, int8_t, int8_t>(args, qp); }
},
{
    GemmMethod::GEMM_INTERLEAVED,
    "a64_gemm_s8_4x4",
    nullptr,
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new GemmInterleavedQuantized<cls_a64_gemm_s8_4x4, int8_t, int8_t>(args, qp); }
},
{
    GemmMethod::QUANTIZE_WRAPPER,
    "quantized_wrapper",
    [](const GemmArgs &args, const Requantize32 &) { return !args._indirect_input; },
    nullptr,
    [](const GemmArgs &args, const Requantize32 &qp) { return new QuantizeWrapper<int8_t, int8_t, int32_t>(args, qp); }
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
const GemmImplementation<int8_t, int8_t, Requantize32> *gemm_implementation_list<int8_t, int8_t, Requantize32>() {
    return gemm_qint8_methods;
}

template UniqueGemmCommon<int8_t, int8_t> gemm<int8_t, int8_t, Requantize32>(const GemmArgs &args, const Requantize32 &os);
template KernelDescription get_gemm_method<int8_t, int8_t, Requantize32>(const GemmArgs &args, const Requantize32 &os);
template std::vector<KernelDescription> get_compatible_kernels<int8_t, int8_t, Requantize32>(const GemmArgs &args, const Requantize32 &os);

} // namespace arm_gemm

#endif // __aarch64__
