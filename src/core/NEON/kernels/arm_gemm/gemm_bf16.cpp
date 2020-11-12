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
#include "arm_gemm.hpp"
#include "bfloat.hpp"
#include "gemm_common.hpp"
#include "gemm_hybrid.hpp"
#include "gemm_hybrid_indirect.hpp"
#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"
#include "gemv_batched.hpp"
#include "gemv_pretransposed.hpp"

#include "kernels/a64_hybrid_bf16fp32_dot_6x16.hpp"
#include "kernels/a64_interleaved_bf16fp32_dot_8x12.hpp"
#include "kernels/a64_interleaved_bf16fp32_mmla_8x12.hpp"
#include "kernels/a64_sgemm_8x12.hpp"
#include "kernels/a32_sgemm_8x6.hpp"
#include "kernels/sve_interleaved_bf16fp32_dot_8x3VL.hpp"
#include "kernels/sve_interleaved_bf16fp32_mmla_8x3VL.hpp"
#include "kernels/sve_hybrid_bf16fp32_dot_6x4VL.hpp"

namespace arm_gemm {

static const GemmImplementation<bfloat16, float> gemm_bf16_methods[] =
{
#ifdef V8P6_BF
#ifdef __ARM_FEATURE_SVE
{ // gemm_bf16_interleaved
    GemmMethod::GEMM_INTERLEAVED,
    "sve_interleaved_bf16fp32_mmla_8x3VL",
    [](const GemmArgs &args) { return (args._Ksize>4); },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_bf16fp32_mmla_8x3VL, bfloat16, float>(args); }
},
{
    GemmMethod::GEMM_HYBRID,
    "sve_hybrid_bf16fp32_dot_6x4VL",
    nullptr,
    [](const GemmArgs &args) { return ((args._Ksize <= 128) && (args._Nsize <= 128)); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_bf16fp32_dot_6x4VL, bfloat16, float>(args); }
},
{ // gemm_bf16_interleaved
    GemmMethod::GEMM_INTERLEAVED,
    "sve_interleaved_bf16fp32_dot_8x3VL",
    [](const GemmArgs &args) { return (args._Ksize>2); },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_bf16fp32_dot_8x3VL, bfloat16, float>(args); }
},
# endif // SVE
{ // gemm_bf16_interleaved
    GemmMethod::GEMM_INTERLEAVED,
    "a64_interleaved_bf16fp32_mmla_8x12",
    [](const GemmArgs &args) { return (args._Ksize>4); },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_interleaved_bf16fp32_mmla_8x12, bfloat16, float>(args); }
},
{
    GemmMethod::GEMM_HYBRID,
    "a64_hybrid_bf16fp32_dot_6x16",
    nullptr,
    nullptr,
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_a64_hybrid_bf16fp32_dot_6x16, bfloat16, float>(args); }
},
{ // gemm_bf16_interleaved
    GemmMethod::GEMM_INTERLEAVED,
    "a64_interleaved_bf16fp32_dot_8x12",
    [](const GemmArgs &args) { return (args._Ksize>2); },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_interleaved_bf16fp32_dot_8x12, bfloat16, float>(args); }
},
#endif // V8P6_BF
#ifdef __aarch64__
{
    GemmMethod::GEMM_INTERLEAVED,
    "a64_sgemm_8x12",
    nullptr,
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_sgemm_8x12, bfloat16, float>(args); }
},
#elif defined(__arm__)
{
    GemmMethod::GEMM_INTERLEAVED,
    "sgemm_8x6",
    nullptr,
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<sgemm_8x6, bfloat16, float>(args); }
},
#else
# error "Unknown Architecture"
#endif
{
    GemmMethod::DEFAULT,
    "",
    nullptr,
    nullptr,
    nullptr
}
};

template<>
const GemmImplementation<bfloat16, float> *gemm_implementation_list<bfloat16, float>() {
    return gemm_bf16_methods;
}

/* Explicitly instantiate the external functions for these types. */
template UniqueGemmCommon<bfloat16, float> gemm<bfloat16, float, Nothing>(const GemmArgs &args, const Nothing &);
template KernelDescription get_gemm_method<bfloat16, float, Nothing>(const GemmArgs &args, const Nothing &);
template std::vector<KernelDescription> get_compatible_kernels<bfloat16, float, Nothing>(const GemmArgs &args, const Nothing &);

} // namespace arm_gemm
