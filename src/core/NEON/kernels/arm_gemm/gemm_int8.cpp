/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "gemm_common.hpp"
#include "gemm_hybrid.hpp"
#include "gemm_hybrid_indirect.hpp"
#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"

#include "kernels/a64_gemm_s16_8x12.hpp"
#include "kernels/a64_gemm_s8_8x12.hpp"
#include "kernels/a64_gemm_s8_4x4.hpp"
#include "kernels/a64_hybrid_s8s32_dot_6x16.hpp"
#include "kernels/a64_interleaved_s8s32_mmla_8x12.hpp"
#include "kernels/a64_smallK_hybrid_s8s32_dot_6x4.hpp"
#include "kernels/a64_smallK_hybrid_s8s32_dot_8x4.hpp"

#include "kernels/sve_hybrid_s8s32_dot_6x4VL.hpp"
#include "kernels/sve_interleaved_s8s32_dot_8x3VL.hpp"
#include "kernels/sve_interleaved_s8s32_mmla_8x3VL.hpp"
#include "kernels/sve_smallK_hybrid_s8s32_dot_8x1VL.hpp"

namespace arm_gemm {

static const GemmImplementation<int8_t, int32_t> gemm_s8_methods[] = {
#ifdef __ARM_FEATURE_SVE
#ifdef MMLA_INT8
{
    GemmMethod::GEMM_INTERLEAVED,
    "sve_interleaved_s8s32_mmla_8x3VL",
    [](const GemmArgs &args) { return (args._Ksize>8); },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_s8s32_mmla_8x3VL, int8_t, int32_t>(args); }
},
#endif
{
    GemmMethod::GEMM_HYBRID,
    "sve_smallK_hybrid_s8s32_dot_8x1VL",
    [](const GemmArgs &args) { return args._Ksize<=64 && !args._indirect_input; },
    nullptr,
    [](const GemmArgs &args) { return new GemmHybrid<cls_sve_smallK_hybrid_s8s32_dot_8x1VL, int8_t, int32_t>(args); }
},
{
    GemmMethod::GEMM_HYBRID,
    "sve_hybrid_s8s32_dot_6x4VL",
    [](const GemmArgs &args) { return args._Ksize>=16; },
    [](const GemmArgs &args) { return ((args._Ksize <= 128) && (args._Nsize <= 128)) || ((args._nmulti > 1) && ((args._Msize / args._maxthreads) < 8)); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_s8s32_dot_6x4VL, int8_t, int32_t>(args); }
},
{
    GemmMethod::GEMM_INTERLEAVED,
    "sve_interleaved_s8s32_dot_8x3VL",
    [](const GemmArgs &args) { return (args._Ksize>4); },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_s8s32_dot_8x3VL, int8_t, int32_t>(args); }
},
#endif // SVE
#ifdef MMLA_INT8
{
    GemmMethod::GEMM_INTERLEAVED,
    "a64_interleaved_s8s32_mmla_8x12",
    [](const GemmArgs &args) { return (args._Ksize>8); },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_interleaved_s8s32_mmla_8x12, int8_t, int32_t>(args); }
},
#endif
{
    GemmMethod::GEMM_HYBRID,
    "a64_smallK_hybrid_s8s32_dot_8x4",
    [](const GemmArgs &args) { return args._ci->has_dotprod() && (args._Nsize % 4 == 0) && (args._Ksize<=32) && !args._indirect_input; },
    nullptr,
    [](const GemmArgs &args) { return new GemmHybrid<cls_a64_smallK_hybrid_s8s32_dot_8x4, int8_t, int32_t>(args); }
},
{
    GemmMethod::GEMM_HYBRID,
    "a64_smallK_hybrid_s8s32_dot_6x4",
    [](const GemmArgs &args) { return args._ci->has_dotprod() && (args._Nsize % 4 == 0) && (args._Ksize>32) && (args._Ksize<=64) && !args._indirect_input; },
    nullptr,
    [](const GemmArgs &args) { return new GemmHybrid<cls_a64_smallK_hybrid_s8s32_dot_6x4, int8_t, int32_t>(args); }
},
{
    GemmMethod::GEMM_INTERLEAVED,
    "a64_gemm_s16_8x12",
    nullptr,
    [](const GemmArgs &args) { return args._ci->get_cpu_model() == CPUModel::A53 && args._Ksize>4; },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_gemm_s16_8x12, int8_t, int32_t>(args); },
},
{
    GemmMethod::GEMM_HYBRID,
    "a64_hybrid_s8s32_dot_6x16",
    [](const GemmArgs &args) { return args._ci->has_dotprod(); },
    [](const GemmArgs &args) { return args._Nsize<=256 && args._Ksize>128; },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_a64_hybrid_s8s32_dot_6x16, int8_t, int32_t>(args); }
},
{
    GemmMethod::GEMM_INTERLEAVED,
    "a64_gemm_s8_8x12",
    [](const GemmArgs &args) { return args._ci->has_dotprod(); },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_gemm_s8_8x12, int8_t, int32_t>(args); }
},
{
    GemmMethod::GEMM_INTERLEAVED,
    "a64_gemm_s8_4x4",
    nullptr,
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_gemm_s8_4x4, int8_t, int32_t>(args); }
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
const GemmImplementation<int8_t, int32_t> *gemm_implementation_list<int8_t, int32_t>() {
    return gemm_s8_methods;
}

/* Explicitly instantiate the external functions for these types. */
template UniqueGemmCommon<int8_t, int32_t> gemm<int8_t, int32_t, Nothing>(const GemmArgs &args, const Nothing &);
template KernelDescription get_gemm_method<int8_t, int32_t, Nothing>(const GemmArgs &args, const Nothing &);
template std::vector<KernelDescription> get_compatible_kernels<int8_t, int32_t, Nothing> (const GemmArgs &args, const Nothing &);

} // namespace arm_gemm

#endif // __aarch64__
