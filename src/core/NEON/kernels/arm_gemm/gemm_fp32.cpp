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
#include "arm_gemm.hpp"
#include "gemm_common.hpp"
#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"
#include "gemm_native.hpp"
#include "gemv_batched.hpp"
#include "gemv_native_transposed.hpp"
#include "gemv_pretransposed.hpp"

#include "kernels/a64_sgemm_12x8.hpp"
#include "kernels/a32_sgemm_8x6.hpp"
#include "kernels/a64_sgemv_trans.hpp"
#include "kernels/a64_sgemv_pretransposed.hpp"
#include "kernels/a64_sgemm_native_16x4.hpp"

namespace arm_gemm {

#ifdef __aarch64__
// SGEMM implementations for AArch64

// Pretransposed GEMV
class GemmImpl_sgemm_gemv_pretransposed : public GemmImplementation<float, float> {
public:
    bool is_supported(const GemmArgs<float> &args) override {
        return (args._Msize==1 && args._alpha==1.0f && args._pretransposed_hint && args._nbatches==1);
    }

    UniqueGemmCommon<float, float> instantiate(const GemmArgs<float> &args) override {
        return UniqueGemmCommon<float, float> (new GemvPretransposed<sgemv_pretransposed, float, float>(args._ci, args._Nsize, args._Ksize, args._nmulti, args._trB, args._beta));
    }

    GemmImpl_sgemm_gemv_pretransposed() : GemmImplementation<float, float>(GemmMethod::GEMV_PRETRANSPOSED) { }
};

// Native GEMV
class GemmImpl_sgemm_gemv_native_transposed : public GemmImplementation<float, float> {
public:
    bool is_supported(const GemmArgs<float> &args) override {
        return (args._Msize==1 && args._alpha==1.0f && !args._trA && !args._trB && args._nbatches==1);
    }

    UniqueGemmCommon<float, float> instantiate(const GemmArgs<float> &args) override {
        return UniqueGemmCommon<float, float> (new GemvNativeTransposed<sgemv_trans, float, float>(args._ci, args._Nsize, args._Ksize, args._nmulti, args._beta));
    }

    GemmImpl_sgemm_gemv_native_transposed() : GemmImplementation<float, float>(GemmMethod::GEMV_NATIVE_TRANSPOSED) { }
};

// Native GEMM
class GemmImpl_sgemm_gemm_native : public GemmImplementation<float, float> {
public:
    bool is_supported(const GemmArgs<float> &args) override {
        return (args._Ksize>4 && (args._Nsize % 16)==0 && args._alpha==1.0f && !args._trA && !args._trB);
    }

    bool is_recommended(const GemmArgs<float> &args) override {
        return ((args._Ksize <= 128) && (args._Nsize <= 128)) || ((args._nmulti > 1) && ((args._Msize / args._maxthreads) < 8));
    }

    UniqueGemmCommon<float, float> instantiate(const GemmArgs<float> &args) override {
        return UniqueGemmCommon<float, float> (new GemmNative<sgemm_native_16x4, float, float>(args._ci, args._Msize, args._Nsize, args._Ksize, args._nbatches, args._nmulti, args._beta));
    }

    GemmImpl_sgemm_gemm_native() : GemmImplementation<float, float>(GemmMethod::GEMM_NATIVE) { }
};
#endif // __aarch64__

// Interleaved GEMM
class GemmImpl_sgemm_gemm_interleaved : public GemmImplementation<float, float> {
public:
    UniqueGemmCommon<float, float> instantiate(const GemmArgs<float> &args) override {
#ifdef __aarch64__
        return UniqueGemmCommon<float, float> (new GemmInterleaved<sgemm_12x8, float, float>(args));
#elif defined(__arm__)
        return UniqueGemmCommon<float, float> (new GemmInterleaved<sgemm_8x6, float, float>(args));
#else
# error Unknown Architecture.
#endif
    }

    GemmImpl_sgemm_gemm_interleaved() : GemmImplementation<float, float>(GemmMethod::GEMM_INTERLEAVED) { }
};

static GemmImpl_gemv_batched<float, float> gemv_batched_impl{};
#ifdef __aarch64__
static GemmImpl_sgemm_gemv_pretransposed sgemm_gemv_pretransposed_impl{};
static GemmImpl_sgemm_gemv_native_transposed sgemm_gemv_native_transposed_impl{};
static GemmImpl_sgemm_gemm_native sgemm_gemm_native_impl{};
#endif
static GemmImpl_sgemm_gemm_interleaved sgemm_gemm_interleaved_impl{};

/* List of implementations (order matters) */
static std::vector<GemmImplementation<float, float> *> SGemmMethods = {
    &gemv_batched_impl,
#ifdef __aarch64__
    &sgemm_gemv_pretransposed_impl,
    &sgemm_gemv_native_transposed_impl,
    &sgemm_gemm_native_impl,
#endif
    &sgemm_gemm_interleaved_impl
};

/* Templated function to return this list. */
template<>
std::vector<GemmImplementation<float, float> *> &gemm_implementation_list<float, float>() {
    return SGemmMethods;
}

/* Explicitly instantiate the external functions for these types. */
template UniqueGemmCommon<float, float> gemm<float, float>(GemmArgs<float> &args, GemmConfig *cfg);
template GemmMethod get_gemm_method<float, float>(GemmArgs<float> &args);
template bool method_is_compatible<float, float>(GemmMethod method, GemmArgs<float> &args);

} // namespace arm_gemm
