/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#pragma once

#include <memory>
#include <cstring>

#include "arm_gemm_local.hpp"
#include "gemm_common.hpp"

namespace arm_gemm {

enum class GemmMethod
{
    DEFAULT,
    GEMV_BATCHED,
    GEMV_PRETRANSPOSED,
    GEMV_NATIVE_TRANSPOSED,
    GEMM_NATIVE,
    GEMM_HYBRID,
    GEMM_INTERLEAVED
};


struct KernelDescription
{
    GemmMethod   method = GemmMethod::DEFAULT;
    std::string  name   = "";

    KernelDescription(GemmMethod m, std::string n) : method(m), name(n) { }
    KernelDescription() { }
};

struct GemmConfig
{
    GemmMethod   method           = GemmMethod::DEFAULT;
    std::string  filter           = "";
    unsigned int inner_block_size = 0;
    unsigned int outer_block_size = 0;

    GemmConfig(GemmMethod method) : method(method) { }
    GemmConfig() { }
};

template<typename T>
struct GemmArgs
{
public:
    const CPUInfo    *_ci;
    unsigned int      _Msize;
    unsigned int      _Nsize;
    unsigned int      _Ksize;
    unsigned int      _nbatches;
    unsigned int      _nmulti;
    bool              _trA;
    bool              _trB;
    T                 _alpha;
    T                 _beta;
    int               _maxthreads;
    bool              _pretransposed_hint;
    const GemmConfig *_cfg;

    GemmArgs(const CPUInfo *ci, const unsigned int M, const unsigned int N,
             const unsigned int K, const unsigned int nbatches,
             const unsigned int nmulti, const bool trA, const bool trB,
             const T alpha, const T beta, const int maxthreads,
             const bool pretransposed_hint, const GemmConfig *cfg=nullptr ) :
            _ci(ci), _Msize(M), _Nsize(N), _Ksize(K), _nbatches(nbatches), _nmulti(nmulti),
            _trA(trA), _trB(trB), _alpha(alpha), _beta(beta), _maxthreads(maxthreads),
            _pretransposed_hint(pretransposed_hint), _cfg(cfg)
    {
    }
};

template<typename Top, typename Tret>
using UniqueGemmCommon = std::unique_ptr<GemmCommon<Top, Tret> >;

/* Low level API calls.
 * These are implemented as 'GemmArgs' versions, or with the arguments explicitly listed. */

/* method_is_compatible(): Can a GEMM of the templated types with the
 * provided parameters be provided using the supplied method?  */

template<typename Top, typename Tret>
bool method_is_compatible(GemmMethod method, const GemmArgs<Tret> &args);

template<typename Top, typename Tret>
bool method_is_compatible(GemmMethod method, const CPUInfo &ci,
                          const unsigned int M, const unsigned int N, const unsigned int K,
                          const unsigned int nbatches, const unsigned int nmulti,
                          const bool trA, const bool trB, const Tret alpha, const Tret beta,
                          const int maxthreads, const bool pretransposed_hint)
{
    GemmArgs<Tret> args(&ci, M, N, K, nbatches, nmulti, trA, trB, alpha, beta, maxthreads, pretransposed_hint);

    return method_is_compatible<Top, Tret>(method, args);
}

/* get_gemm_method(): Given the templated types and provided parameters,
 * which is the preferred method to implement this GEMM?  */
template<typename Top, typename Tret>
KernelDescription get_gemm_method(const GemmArgs<Tret> &args);

template<typename Top, typename Tret>
KernelDescription get_gemm_method(const CPUInfo &ci,
                                  const unsigned int M, const unsigned int N, const unsigned int K,
                                  const unsigned int nbatches, const unsigned int nmulti,
                                  const bool trA, const bool trB, const Tret alpha, const Tret beta,
                                  const int maxthreads, const bool pretransposed_hint)
{
    GemmArgs<Tret> args(&ci, M, N, K, nbatches, nmulti, trA, trB, alpha, beta, maxthreads, pretransposed_hint);

    return get_gemm_method<Top, Tret>(args);
}

template<typename Top, typename Tret>
UniqueGemmCommon<Top, Tret> gemm(const GemmArgs<Tret> &args);

/** Request an object to process a GEMM.
 *
 * @param[in]  ci                 Describes CPU properties.
 * @param[in]  M                  Rows in output matrix C (and input matrix A).
 * @param[in]  N                  Columns in output matrix C (and input matrix B).
 * @param[in]  K                  Columns of input matrix A (= rows of input matrix B).
 * @param[in]  nbatches           Number of "batched" GEMMs (unique A and C, shared B).
 * @param[in]  nmulti             Number of "multi" GEMMs (unique A, B and C).
 * @param[in]  trA                Does A tensor has rows and columns transposed?
 * @param[in]  trB                Does B tensor has rows and columns transposed?
 * @param[in]  alpha              Scalar multiplier to apply to AB matrix product.
 * @param[in]  beta               Scalar multiplier to apply to input C matrix before adding product.
 * @param[in]  maxthreads         Maximum (and default) number of threads that will call execute method.
 * @param[in]  pretransposed_hint Can the B tensor can be pretransposed (ie shared across invocations)?
 * @param[in]  cfg                (optional) configuration parameters
 */
template<typename Top, typename Tret>
UniqueGemmCommon<Top, Tret> gemm(const CPUInfo &ci,
                                 const unsigned int M, const unsigned int N, const unsigned int K,
                                 const unsigned int nbatches, const unsigned int nmulti,
                                 const bool trA, const bool trB, const Tret alpha, const Tret beta,
                                 const int maxthreads, const bool pretransposed_hint, GemmConfig *cfg=nullptr)
{
    GemmArgs<Tret> args(&ci, M, N, K, nbatches, nmulti, trA, trB, alpha, beta, maxthreads, pretransposed_hint, cfg);

    return gemm<Top, Tret>(args);
}

template<typename Top, typename Tret>
std::vector<std::string> get_compatible_kernels(const GemmArgs<Tret> &args);

template<typename Top, typename Tret>
std::vector<std::string> get_compatible_kernels(const CPUInfo &ci,
                                                const unsigned int M, const unsigned int N, const unsigned int K,
                                                const unsigned int nbatches, const unsigned int nmulti,
                                                const bool trA, const bool trB, const Tret alpha, const Tret beta,
                                                const int maxthreads, const bool pretransposed_hint)
{
    GemmArgs<Tret> args(&ci, M, N, K, nbatches, nmulti, trA, trB, alpha, beta, maxthreads, pretransposed_hint);

    return get_compatible_kernels<Top, Tret>(args);
}

} // namespace arm_gemm
