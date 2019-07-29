/*
 * Copyright (c) 2019 Arm Limited.
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

#include "kernels/a64_hybrid_u8u32_dot_16x4.hpp"
#include "kernels/a64_smallK_hybrid_u8u32_dot_4x6.hpp"
#include "kernels/a64_smallK_hybrid_u8u32_dot_4x8.hpp"
#include "kernels/sve_hybrid_u8u32_dot_4VLx4.hpp"
#include "kernels/sve_smallK_hybrid_u8u32_dot_1VLx8.hpp"

#include "gemm_hybrid_quantized.hpp"
#include "quantize_wrapper.hpp"

namespace arm_gemm {

static const GemmImplementation<uint8_t, uint8_t, ARequantizeLayer32> gemm_quint8_methods[] =
{
#ifdef __ARM_FEATURE_SVE
{
    GemmMethod::GEMM_HYBRID_QUANTIZED,
    "smallK_hybrid_u8u32_dot_1VLx8",
    [](const GemmArgs<uint8_t> &args, const ARequantizeLayer32 &) { return args._Ksize<=64 && args._alpha==1 && args._beta==0 && !args._trA && args._pretransposed_hint; },
    nullptr,
    [](const GemmArgs<uint8_t> &args, const ARequantizeLayer32 &qp) { return new GemmHybridQuantized<smallK_hybrid_u8u32_dot_1VLx8, uint8_t, uint8_t>(args, qp); }
},
{
    GemmMethod::GEMM_HYBRID_QUANTIZED,
    "hybrid_u8u32_dot_4VLx4",
    [](const GemmArgs<uint8_t> &args, const ARequantizeLayer32 &) { return args._Ksize>=16 && args._alpha==1 && !args._trA && !args._trB && args._pretransposed_hint; },
    [](const GemmArgs<uint8_t> &args, const ARequantizeLayer32 &) { return ((args._Ksize <= 128) && (args._Nsize <= 128)) || ((args._nmulti > 1) && ((args._Msize / args._maxthreads) < 8)); },
    [](const GemmArgs<uint8_t> &args, const ARequantizeLayer32 &qp) { return new GemmHybridQuantized<hybrid_u8u32_dot_4VLx4, uint8_t, uint8_t>(args, qp); }
},
#endif
{
    GemmMethod::GEMM_HYBRID_QUANTIZED,
    "smallK_hybrid_u8u32_dot_4x8",
    [](const GemmArgs<uint8_t> &args, const ARequantizeLayer32 &) { return args._ci->has_dotprod() && (args._Nsize % 4 == 0) && (args._Ksize<=32) && args._alpha==1 && args._beta==0 && !args._trA && args._pretransposed_hint; },
    nullptr,
    [](const GemmArgs<uint8_t> &args, const ARequantizeLayer32 &qp) { return new GemmHybridQuantized<smallK_hybrid_u8u32_dot_4x8, uint8_t, uint8_t>(args, qp); }
},
{
    GemmMethod::GEMM_HYBRID_QUANTIZED,
    "smallK_hybrid_u8u32_dot_4x6",
    [](const GemmArgs<uint8_t> &args, const ARequantizeLayer32 &) { return args._ci->has_dotprod() && (args._Nsize % 4 == 0) && (args._Ksize>32) && (args._Ksize<=64) && args._alpha==1 && args._beta==0 && !args._trA && args._pretransposed_hint; },
    nullptr,
    [](const GemmArgs<uint8_t> &args, const ARequantizeLayer32 &qp) { return new GemmHybridQuantized<smallK_hybrid_u8u32_dot_4x6, uint8_t, uint8_t>(args, qp); }
},
{
    GemmMethod::GEMM_HYBRID_QUANTIZED,
    "hybrid_u8u32_dot_16x4",
    [](const GemmArgs<uint8_t> &args, const ARequantizeLayer32 &) { return args._ci->has_dotprod() && args._Ksize>=16 && !args._trA && !args._trB && args._pretransposed_hint; },
    [](const GemmArgs<uint8_t> &args, const ARequantizeLayer32 &) { return args._Nsize<=256 && args._Ksize>128; },
    [](const GemmArgs<uint8_t> &args, const ARequantizeLayer32 &qp) { return new GemmHybridQuantized<hybrid_u8u32_dot_16x4, uint8_t, uint8_t>(args, qp); }
},
{
    GemmMethod::QUANTIZE_WRAPPER,
    "quantized_wrapper",
    nullptr,
    nullptr,
    [](const GemmArgs<uint8_t> &args, const ARequantizeLayer32 &qp) { return new QuantizeWrapper<uint8_t, uint8_t, uint32_t>(args, qp); }
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
const GemmImplementation<uint8_t, uint8_t, ARequantizeLayer32> *gemm_implementation_list<uint8_t, uint8_t, ARequantizeLayer32>() {
    return gemm_quint8_methods;
}

template UniqueGemmCommon<uint8_t, uint8_t> gemm<uint8_t, uint8_t, ARequantizeLayer32>(const GemmArgs<uint8_t> &args, const ARequantizeLayer32 &os);
template KernelDescription get_gemm_method<uint8_t, uint8_t, ARequantizeLayer32>(const GemmArgs<uint8_t> &args, const ARequantizeLayer32 &os);
template std::vector<KernelDescription> get_compatible_kernels<uint8_t, uint8_t, ARequantizeLayer32>(const GemmArgs<uint8_t> &args, const ARequantizeLayer32 &os);

} // namespace arm_gemm

#endif // __aarch64__
