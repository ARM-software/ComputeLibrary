/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"

#include "kernels/a64_hgemm_24x8.hpp"
#include "kernels/a64_sgemm_12x8.hpp"
#include "kernels/a32_sgemm_8x6.hpp"

namespace arm_gemm {

#ifdef __aarch64__

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(FP16_KERNELS)
class GemmImpl_gemm_fp16_interleaved_fp16 : public GemmImplementation<__fp16, __fp16> {
public:
#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    bool is_supported(const GemmArgs<__fp16> &args) override {
        return args._ci->has_fp16();
    }
#endif

    UniqueGemmCommon<__fp16, __fp16> instantiate(const GemmArgs<__fp16> &args) override {
        return UniqueGemmCommon<__fp16, __fp16>(new GemmInterleaved<hgemm_24x8, __fp16, __fp16>(args));
    }

    GemmImpl_gemm_fp16_interleaved_fp16() : GemmImplementation<__fp16, __fp16>(GemmMethod::GEMM_INTERLEAVED_FP16) { }
};
#endif

#endif // __aarch64__

class GemmImpl_gemm_fp16_interleaved : public GemmImplementation<__fp16, __fp16> {
public:
    UniqueGemmCommon<__fp16, __fp16> instantiate(const GemmArgs<__fp16> &args) override {
#ifdef __aarch64__
        return UniqueGemmCommon<__fp16, __fp16>(new GemmInterleaved<sgemm_12x8, __fp16, __fp16>(args));
#elif defined(__arm__)
        return UniqueGemmCommon<__fp16, __fp16>(new GemmInterleaved<sgemm_8x6, __fp16, __fp16>(args));
#else
# error Unknown Architecture
#endif
    }

    GemmImpl_gemm_fp16_interleaved() : GemmImplementation<__fp16, __fp16>(GemmMethod::GEMM_INTERLEAVED) { }
};

#if defined(__aarch64__) && (defined(__ARM_FEATURE_VECTOR_ARITHMETIC) || defined(FP16_KERNELS))
static GemmImpl_gemm_fp16_interleaved_fp16 gemm_fp16_interleaved_fp16_impl{};
#endif
static GemmImpl_gemm_fp16_interleaved gemm_fp16_interleaved_impl{};

static std::vector<GemmImplementation<__fp16, __fp16> *> gemm_fp16_methods = {
#if defined(__aarch64__) && (defined(__ARM_FEATURE_VECTOR_ARITHMETIC) || defined(FP16_KERNELS))
    &gemm_fp16_interleaved_fp16_impl,
#endif
    &gemm_fp16_interleaved_impl
};

template<>
std::vector<GemmImplementation<__fp16, __fp16> *> &gemm_implementation_list<__fp16, __fp16>() {
    return gemm_fp16_methods;
}

/* Explicitly instantiate the external functions for these types. */
template UniqueGemmCommon<__fp16, __fp16> gemm<__fp16, __fp16>(GemmArgs<__fp16> &args, GemmConfig *cfg);
template GemmMethod get_gemm_method<__fp16, __fp16>(GemmArgs<__fp16> &args);
template bool method_is_compatible<__fp16, __fp16>(GemmMethod method, GemmArgs<__fp16> &args);

} // namespace arm_gemm

#endif // __ARM_FP16_ARGS
