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
#include "arm_gemm.hpp"
#include "gemm_common.hpp"
#include "gemm_hybrid.hpp"
#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"
#include "gemm_native.hpp"
#include "gemv_batched.hpp"
#include "gemv_native_transposed.hpp"
#include "gemv_pretransposed.hpp"

#include "kernels/a32_sgemm_8x6.hpp"
#include "kernels/a64_sgemm_12x8.hpp"
#include "kernels/a64_sgemm_native_16x4.hpp"
#include "kernels/a64_sgemm_nativeA_pretransposeB_16x4.hpp"
#include "kernels/a64_sgemv_pretransposed.hpp"
#include "kernels/a64_sgemv_trans.hpp"

#include "kernels/sve_hybrid_fp32_mla_4VLx4.hpp"
#include "kernels/sve_interleaved_fp32_mla_3VLx8.hpp"
#include "kernels/sve_native_fp32_mla_4VLx4.hpp"
#include "kernels/sve_smallK_fp32_mla_1VLx4.hpp"
#include "kernels/sve_smallK_hybrid_fp32_mla_1VLx4.hpp"

namespace arm_gemm {

static const GemmImplementation<float, float> gemm_fp32_methods[] =
{
{
    GemmMethod::GEMV_BATCHED,
    "gemv_batched",
    [](const GemmArgs<float> &args) { return (args._Msize==1) && (args._nbatches>1); },
    nullptr,
    [](const GemmArgs<float> &args) { return new GemvBatched<float, float>(args); }
},
#ifdef __aarch64__
{
    GemmMethod::GEMV_PRETRANSPOSED,
    "sgemv_pretransposed",
    [](const GemmArgs<float> &args) { return (args._Msize==1 && args._alpha==1.0f && args._pretransposed_hint && args._nbatches==1); },
    nullptr,
    [](const GemmArgs<float> &args) { return new GemvPretransposed<sgemv_pretransposed, float, float>(args); }
},
{
    GemmMethod::GEMV_NATIVE_TRANSPOSED,
    "sgemv_trans",
    [](const GemmArgs<float> &args) { return (args._Msize==1 && args._alpha==1.0f && !args._trA && !args._trB && args._nbatches==1); },
    nullptr,
    [](const GemmArgs<float> &args) { return new GemvNativeTransposed<sgemv_trans, float, float>(args); }
},

#ifdef __ARM_FEATURE_SVE
        // SVE smallk / native / hybrid methods
{
    GemmMethod::GEMM_HYBRID,
    "smallK_hybrid_fp32_mla_1VLx4",
    [](const GemmArgs<float> &args) { return (args._Ksize <= 24) && !args._trA && args._alpha==1.0f && args._pretransposed_hint; },
    nullptr,
    [](const GemmArgs<float> &args) { return new GemmHybrid<smallK_hybrid_fp32_mla_1VLx4, float, float>(args); }
},
{
    GemmMethod::GEMM_HYBRID,
    "hybrid_fp32_mla_4VLx4",
    [](const GemmArgs<float> &args) { return (args._Ksize >= 4) && (args._alpha == 1.0f) && !args._trA && args._pretransposed_hint; },
    [](const GemmArgs<float> &args) { return ((args._Ksize <= 256) && (args._Nsize <= 256)) || ((args._nmulti > 1) && ((args._Msize / args._maxthreads) < 8)); },
    [](const GemmArgs<float> &args) { return new GemmHybrid<hybrid_fp32_mla_4VLx4, float, float>(args); }
},
{
    GemmMethod::GEMM_NATIVE,
    "smallK_fp32_mla_1VLx4",
    [](const GemmArgs<float> &args) { return (args._Ksize <= 24) && !args._trA && !args._trB && args._alpha==1.0f; },
    nullptr,
    [](const GemmArgs<float> &args) { return new GemmNative<smallK_fp32_mla_1VLx4, float, float>(args); }
},
{
    GemmMethod::GEMM_NATIVE,
    "native_fp32_mla_4VLx4",
    [](const GemmArgs<float> &args) { return (args._Ksize>4 && args._alpha==1.0f && !args._trA && !args._trB); },
    [](const GemmArgs<float> &args) { return ((args._Ksize <= 128) && (args._Nsize <= 128)) || ((args._nmulti > 1) && ((args._Msize / args._maxthreads) < 8)); },
    [](const GemmArgs<float> &args) { return new GemmNative<native_fp32_mla_4VLx4, float, float>(args); }
},
#endif // __ARM_FEATURE_SVE

// NEON native / hybrid methods
{
    GemmMethod::GEMM_HYBRID,
    "sgemm_nativeA_pretransposeB_16x4",
    [](const GemmArgs<float> &args) { return (args._Ksize >= 4) && (args._alpha == 1.0f) && !args._trA && args._pretransposed_hint; },
    [](const GemmArgs<float> &args) { return ((args._Ksize <= 256) && (args._Nsize <= 256)) || ((args._nmulti > 1) && ((args._Msize / args._maxthreads) < 8)); },
    [](const GemmArgs<float> &args) { return new GemmHybrid<sgemm_nativeA_pretransposeB_16x4, float, float>(args); }
},
{
    GemmMethod::GEMM_NATIVE,
    "sgemm_native_16x4",
    [](const GemmArgs<float> &args) { return (args._Ksize>4 && (args._Nsize % 16)==0 && args._alpha==1.0f && !args._trA && !args._trB); },
    [](const GemmArgs<float> &args) { return ((args._Ksize <= 128) && (args._Nsize <= 128)) || ((args._nmulti > 1) && ((args._Msize / args._maxthreads) < 8)); },
    [](const GemmArgs<float> &args) { return new GemmNative<sgemm_native_16x4, float, float>(args); }
},

#ifdef __ARM_FEATURE_SVE
        {
    GemmMethod::GEMM_INTERLEAVED,
    "interleaved_fp32_mla_3VLx8",
    [](const GemmArgs<float> &args) { return (args._Ksize>4); },
    nullptr,
    [](const GemmArgs<float> &args) { return new GemmInterleaved<interleaved_fp32_mla_3VLx8, float, float>(args); }
},
#endif // __ARM_FEATURE_SVE
{
    GemmMethod::GEMM_INTERLEAVED,
    "sgemm_12x8",
    nullptr,
    nullptr,
    [](const GemmArgs<float> &args) { return new GemmInterleaved<sgemm_12x8, float, float>(args); }
},
#endif // __aarch64__

#ifdef __arm__
        {
    GemmMethod::GEMM_INTERLEAVED,
    "sgemm_8x6",
    nullptr,
    nullptr,
    [](const GemmArgs<float> &args) { return new GemmInterleaved<sgemm_8x6, float, float>(args); }
},
#endif // __arm__
{
    GemmMethod::DEFAULT,
    "",
    nullptr,
    nullptr,
    nullptr
}
};

/* Templated function to return this list. */
template<>
const GemmImplementation<float, float> *gemm_implementation_list<float, float>() {
    return gemm_fp32_methods;
}

/* Explicitly instantiate the external functions for these types. */
template UniqueGemmCommon<float, float> gemm<float, float>(const GemmArgs<float> &args);
template KernelDescription get_gemm_method<float, float>(const GemmArgs<float> &args);
template bool method_is_compatible<float, float>(GemmMethod method, const GemmArgs<float> &args);
template std::vector<std::string> get_compatible_kernels<float, float> (const GemmArgs<float> &args);

} // namespace arm_gemm