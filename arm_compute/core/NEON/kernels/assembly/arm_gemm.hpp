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
    GEMM_INTERLEAVED,
    QUANTIZE_WRAPPER,
    GEMM_HYBRID_QUANTIZED
};

struct KernelDescription
{
    GemmMethod   method      = GemmMethod::DEFAULT;
    std::string  name        = "";
    bool         is_default  = false;

    KernelDescription(GemmMethod m, std::string n, bool d=false) : method(m), name(n), is_default(d) { }
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

struct ARequantizeLayer32
{
public:
    const int32_t  *bias;
    int32_t         a_offset;
    int32_t         b_offset;
    int32_t         c_offset;
    int32_t         requant_shift;
    int32_t         requant_mul;
    int32_t         minval;
    int32_t         maxval;

    ARequantizeLayer32() = default;

    ARequantizeLayer32(int32_t *b, int32_t ao, int32_t bo, int32_t co, int32_t rs, int32_t rm, int32_t minv, int32_t maxv) :
        bias(b), a_offset(ao), b_offset(bo), c_offset(co), requant_shift(rs), requant_mul(rm), minval(minv), maxval(maxv)
    {
    }
};

struct Nothing
{
};

template<typename Top, typename Tret>
using UniqueGemmCommon = std::unique_ptr<GemmCommon<Top, Tret> >;

/* Low level API calls.
 * These are implemented as 'GemmArgs' versions, or with the arguments explicitly listed. */

/* get_gemm_method(): Given the templated types and provided parameters,
 * which is the preferred method to implement this GEMM?  */
template<typename Top, typename Tret, class OutputStage = Nothing>
KernelDescription get_gemm_method(const GemmArgs<Tret> &args, const OutputStage & ={});

template<typename Top, typename Tret, class OutputStage = Nothing>
UniqueGemmCommon<Top, Tret> gemm(const GemmArgs<Tret> &args, const OutputStage & ={});

template<typename Top, typename Tret, class OutputStage = Nothing>
std::vector<KernelDescription> get_compatible_kernels(const GemmArgs<Tret> &args, const OutputStage & ={});

} // namespace arm_gemm
