/*
 * Copyright (c) 2017-2019 ARM Limited.
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

// This can only be built if the target/compiler supports FP16 arguments.
#ifdef __ARM_FP16_ARGS

#include "arm_gemm.hpp"

#include "gemm_common.hpp"
#include "gemm_hybrid.hpp"
#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"
#include "gemm_native.hpp"

#include "kernels/a32_sgemm_8x6.hpp"
#include "kernels/a64_hgemm_24x8.hpp"
#include "kernels/a64_sgemm_12x8.hpp"
#include "kernels/sve_hybrid_fp16_mla_4VLx4.hpp"
#include "kernels/sve_interleaved_fp16_mla_3VLx8.hpp"
#include "kernels/sve_native_fp16_mla_4VLx4.hpp"

namespace arm_gemm {

static const GemmImplementation<__fp16, __fp16> gemm_fp16_methods[] = {
#if defined(__ARM_FEATURE_SVE)
{
    GemmMethod::GEMM_HYBRID,
    "hybrid_fp16_mla_4VLx4",
    [](const GemmArgs<__fp16> &args) { return (args._Ksize >= 8) && (args._alpha == 1.0f) && !args._trA && args._pretransposed_hint; },
    [](const GemmArgs<__fp16> &args) { return ((args._Ksize <= 256) && (args._Nsize <= 256)) || ((args._nmulti > 1) && ((args._Msize / args._maxthreads) < 8)); },
    [](const GemmArgs<__fp16> &args) { return new GemmHybrid<hybrid_fp16_mla_4VLx4, __fp16, __fp16>(args); }
},
{
    GemmMethod::GEMM_NATIVE,
    "native_fp16_mla_4VLx4",
    [](const GemmArgs<__fp16> &args) { return (args._Ksize >= 8 && args._alpha==1.0f && !args._trA && !args._trB); },
    [](const GemmArgs<__fp16> &args) { return ((args._Ksize <= 128) && (args._Nsize <= 128)) || ((args._nmulti > 1) && ((args._Msize / args._maxthreads) < 8)); },
    [](const GemmArgs<__fp16> &args) { return new GemmNative<native_fp16_mla_4VLx4, __fp16, __fp16>(args); }
},
{
    GemmMethod::GEMM_INTERLEAVED,
    "interleaved_fp16_mla_3VLx8",
    [](const GemmArgs<__fp16> &args) { return (args._Ksize > 4); },
    nullptr,
    [](const GemmArgs<__fp16> &args) { return new GemmInterleaved<interleaved_fp16_mla_3VLx8, __fp16, __fp16>(args); }
},
#endif

#if defined(__aarch64__) && (defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(FP16_KERNELS))
{
    GemmMethod::GEMM_INTERLEAVED,
    "hgemm_24x8",
#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    [](const GemmArgs<__fp16> &args) { return args._ci->has_fp16(); },
#else
    nullptr,
#endif
    nullptr,
    [](const GemmArgs<__fp16> &args) { return new GemmInterleaved<hgemm_24x8, __fp16, __fp16>(args); }
},
#endif
#ifdef __aarch64__
{
    GemmMethod::GEMM_INTERLEAVED,
    "sgemm_12x8",
    nullptr,
    nullptr,
    [](const GemmArgs<__fp16> &args) { return new GemmInterleaved<sgemm_12x8, __fp16, __fp16>(args); }
},
#elif defined(__arm__)
{
    GemmMethod::GEMM_INTERLEAVED,
    "sgemm_8x6",
    nullptr,
    nullptr,
    [](const GemmArgs<__fp16> &args) { return new GemmInterleaved<sgemm_8x6, __fp16, __fp16>(args); }
},
#else // not AArch64 or AArch32
# error Unknown Architecture
#endif
{
    GemmMethod::DEFAULT,
    "",
    nullptr,
    nullptr,
    nullptr,
}
};

template<>
const GemmImplementation<__fp16, __fp16> *gemm_implementation_list<__fp16, __fp16>() {
    return gemm_fp16_methods;
}

/* Explicitly instantiate the external functions for these types. */
template UniqueGemmCommon<__fp16, __fp16> gemm<__fp16, __fp16>(const GemmArgs<__fp16> &args);
template KernelDescription get_gemm_method<__fp16, __fp16>(const GemmArgs<__fp16> &args);
template bool method_is_compatible<__fp16, __fp16>(GemmMethod method, const GemmArgs<__fp16> &args);
template std::vector<KernelDescription> get_compatible_kernels<__fp16, __fp16> (const GemmArgs<__fp16> &args);

} // namespace arm_gemm

#endif // __ARM_FP16_ARGS
