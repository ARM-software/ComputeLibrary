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
#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"
#include "gemm_interleaved_pretransposed_2d.hpp"

#include "kernels/a64_gemm_s16_12x8.hpp"
#include "kernels/a64_gemm_s8_12x8.hpp"
#include "kernels/a64_gemm_s8_4x4.hpp"
#include "kernels/a64_hybrid_s8s32_dot_16x4.hpp"
#include "kernels/a64_interleaved_s8s32_mmla_12x8.hpp"
#include "kernels/a64_smallK_hybrid_s8s32_dot_4x6.hpp"
#include "kernels/a64_smallK_hybrid_s8s32_dot_4x8.hpp"
#include "kernels/sve_hybrid_s8s32_dot_4VLx4.hpp"
#include "kernels/sve_interleaved_s8s32_dot_3VLx8.hpp"
#include "kernels/sve_interleaved_s8s32_mmla_3VLx8.hpp"
#include "kernels/sve_smallK_hybrid_s8s32_dot_1VLx8.hpp"

namespace arm_gemm {

static const GemmImplementation<int8_t, int32_t> gemm_s8_methods[] = {
#ifdef __ARM_FEATURE_SVE
#ifdef MMLA_INT8
{
    GemmMethod::GEMM_INTERLEAVED,
    "interleaved_s8s32_mmla_3VLx8",
    [](const GemmArgs &args) { return (args._Ksize>8); },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<interleaved_s8s32_mmla_3VLx8, int8_t, int32_t>(args); }
},
#endif
{
    GemmMethod::GEMM_HYBRID,
    "smallK_hybrid_s8s32_dot_1VLx8",
    [](const GemmArgs &args) { return args._Ksize<=64; },
    nullptr,
    [](const GemmArgs &args) { return new GemmHybrid<smallK_hybrid_s8s32_dot_1VLx8, int8_t, int32_t>(args); }
},
{
    GemmMethod::GEMM_HYBRID,
    "hybrid_s8s32_dot_4VLx4",
    [](const GemmArgs &args) { return args._Ksize>=16; },
    [](const GemmArgs &args) { return ((args._Ksize <= 128) && (args._Nsize <= 128)) || ((args._nmulti > 1) && ((args._Msize / args._maxthreads) < 8)); },
    [](const GemmArgs &args) { return new GemmHybrid<hybrid_s8s32_dot_4VLx4, int8_t, int32_t>(args); }
},
{
    GemmMethod::GEMM_INTERLEAVED,
    "interleaved_s8s32_dot_3VLx8",
    [](const GemmArgs &args) { return (args._Ksize>4); },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<interleaved_s8s32_dot_3VLx8, int8_t, int32_t>(args); }
},
#endif
#ifdef MMLA_INT8
{
    GemmMethod::GEMM_INTERLEAVED,
    "interleaved_s8s32_mmla_12x8",
    [](const GemmArgs &args) { return (args._Ksize>8); },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<interleaved_s8s32_mmla_12x8, int8_t, int32_t>(args); }
},
#endif
{
    GemmMethod::GEMM_HYBRID,
    "smallK_hybrid_s8s32_dot_4x8",
    [](const GemmArgs &args) { return args._ci->has_dotprod() && (args._Nsize % 4 == 0) && (args._Ksize<=32); },
    nullptr,
    [](const GemmArgs &args) { return new GemmHybrid<smallK_hybrid_s8s32_dot_4x8, int8_t, int32_t>(args); }
},
{
    GemmMethod::GEMM_HYBRID,
    "smallK_hybrid_s8s32_dot_4x6",
    [](const GemmArgs &args) { return args._ci->has_dotprod() && (args._Nsize % 4 == 0) && (args._Ksize>32) && (args._Ksize<=64); },
    nullptr,
    [](const GemmArgs &args) { return new GemmHybrid<smallK_hybrid_s8s32_dot_4x6, int8_t, int32_t>(args); }
},
{
    GemmMethod::GEMM_HYBRID,
    "hybrid_s8s32_dot_16x4",
    [](const GemmArgs &args) { return args._ci->has_dotprod() && args._Ksize>=16; },
    [](const GemmArgs &args) { return args._Nsize<=256 && args._Ksize>128; },
    [](const GemmArgs &args) { return new GemmHybrid<hybrid_s8s32_dot_16x4, int8_t, int32_t>(args); }
},
{
    GemmMethod::GEMM_INTERLEAVED_2D,
    "gemm_s8_12x8_2d",
    [](const GemmArgs &args) { return args._ci->has_dotprod(); },
    [](const GemmArgs &args) { return (args._maxthreads >= 8) && (args._Msize >= 8) && (args._Nsize >= 8); },
    [](const GemmArgs &args) { return new GemmInterleavedPretransposed2d<gemm_s8_12x8, int8_t, int32_t>(args); }
},
{
    GemmMethod::GEMM_INTERLEAVED,
    "gemm_s8_12x8_1d",
    [](const GemmArgs &args) { return args._ci->has_dotprod(); },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<gemm_s8_12x8, int8_t, int32_t>(args); }
},
{
    GemmMethod::GEMM_INTERLEAVED_2D,
    "gemm_s16_12x8_2d",
    nullptr,
    [](const GemmArgs &args) { return args._ci->get_cpu_model() == CPUModel::A53 && args._Msize > 4 && (args._Msize / args._maxthreads) < 8; },
    [](const GemmArgs &args) { return new GemmInterleavedPretransposed2d<gemm_s16_12x8, int8_t, int32_t>(args); },
},
{
    GemmMethod::GEMM_INTERLEAVED,
    "gemm_s16_12x8_1d",
    nullptr,
    [](const GemmArgs &args) { return args._ci->get_cpu_model() == CPUModel::A53 && args._Msize > 4; },
    [](const GemmArgs &args) { return new GemmInterleaved<gemm_s16_12x8, int8_t, int32_t>(args); },
},
{
    GemmMethod::GEMM_INTERLEAVED_2D,
    "gemm_s8_4x4_2d",
    nullptr,
    [](const GemmArgs &args) { return ((args._maxthreads >= 8) && (args._Msize >= 8) && (args._Nsize >= 8)) ||
                                       ((args._Msize / args._maxthreads) < 4); },
    [](const GemmArgs &args) { return new GemmInterleavedPretransposed2d<gemm_s8_4x4, int8_t, int32_t>(args); }
},
{
    GemmMethod::GEMM_INTERLEAVED,
    "gemm_s8_4x4_1d",
    nullptr,
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<gemm_s8_4x4, int8_t, int32_t>(args); }
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
