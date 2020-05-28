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
#include "gemm_hybrid.hpp"
#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"
#include "gemm_native.hpp"
#include "gemv_batched.hpp"
#include "gemv_native_transposed.hpp"
#include "gemv_pretransposed.hpp"

#include "kernels/a64_interleaved_bf16fp32_dot_12x8.hpp"
#include "kernels/a64_interleaved_bf16fp32_mmla_12x8.hpp"
#include "kernels/a64_sgemm_12x8.hpp"
#include "kernels/a32_sgemm_8x6.hpp"
#include "kernels/sve_interleaved_bf16fp32_dot_3VLx8.hpp"
#include "kernels/sve_interleaved_bf16fp32_mmla_3VLx8.hpp"
#include "kernels/sve_native_bf16fp32_dot_4VLx4.hpp"
#include "kernels/sve_hybrid_bf16fp32_dot_4VLx4.hpp"
#include "kernels/sve_hybrid_bf16fp32_mmla_4VLx4.hpp"
#include "kernels/sve_hybrid_bf16fp32_mmla_6VLx2.hpp"
#include "kernels/sve_hybrid_bf16fp32_mmla_8VLx2.hpp"

#include "bfloat.hpp"

namespace arm_gemm {


static const GemmImplementation<bfloat16, float> gemm_bf16_methods[] =
{
#ifdef V8P6_BF
# ifdef __ARM_FEATURE_SVE
{
    GemmMethod::GEMM_HYBRID,
    "hybrid_bf16fp32_mmla_6VLx2",
    [](const GemmArgs &args) { return (args._Ksize>=8 && !args._trA && args._pretransposed_hint); },
    [](const GemmArgs &args) { return ((args._Msize <= 4) && (args._Nsize <= hybrid_bf16fp32_mmla_6VLx2::out_width())); },
    [](const GemmArgs &args) { return new GemmHybrid<hybrid_bf16fp32_mmla_6VLx2, bfloat16, float>(args); }
},
{
    GemmMethod::GEMM_HYBRID,
    "hybrid_bf16fp32_mmla_8VLx2",
    [](const GemmArgs &args) { return (args._Ksize>=8 && !args._trA && args._pretransposed_hint); },
    [](const GemmArgs &args) { return (args._Msize <= 4); },
    [](const GemmArgs &args) { return new GemmHybrid<hybrid_bf16fp32_mmla_8VLx2, bfloat16, float>(args); }
},
{
    GemmMethod::GEMM_HYBRID,
    "hybrid_bf16fp32_mmla_4VLx4",
    [](const GemmArgs &args) { return (args._Ksize>=8 && !args._trA && args._pretransposed_hint); },
    [](const GemmArgs &args) { return ((args._Ksize <= 128) && (args._Nsize <= 128)); },
    [](const GemmArgs &args) { return new GemmHybrid<hybrid_bf16fp32_mmla_4VLx4, bfloat16, float>(args); }
},
{
    GemmMethod::GEMM_HYBRID,
    "hybrid_bf16fp32_dot_4VLx4",
    [](const GemmArgs &args) { return (args._Ksize>=8 && !args._trA && args._pretransposed_hint); },
    [](const GemmArgs &args) { return ((args._Ksize <= 128) && (args._Nsize <= 128)); },
    [](const GemmArgs &args) { return new GemmHybrid<hybrid_bf16fp32_dot_4VLx4, bfloat16, float>(args); }
},
{ // gemm_bf16_native
    GemmMethod::GEMM_NATIVE,
    "native_bf16fp32_dot_4VLx4",
    [](const GemmArgs &args) { return (args._Ksize>=8 && !args._trA && !args._trB); },
    [](const GemmArgs &args) { return ((args._Ksize <= 128) && (args._Nsize <= 128)); },
    [](const GemmArgs &args) { return new GemmNative<native_bf16fp32_dot_4VLx4, bfloat16, float>(args); }
},
{ // gemm_bf16_interleaved
    GemmMethod::GEMM_INTERLEAVED,
    "interleaved_bf16fp32_mmla_3VLx8",
    [](const GemmArgs &args) { return (args._Ksize>4); },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<interleaved_bf16fp32_mmla_3VLx8, bfloat16, float>(args); }
},
{ // gemm_bf16_interleaved
    GemmMethod::GEMM_INTERLEAVED,
    "interleaved_bf16fp32_dot_3VLx8",
    [](const GemmArgs &args) { return (args._Ksize>2); },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<interleaved_bf16fp32_dot_3VLx8, bfloat16, float>(args); }
},
# endif // SVE
{ // gemm_bf16_interleaved
    GemmMethod::GEMM_INTERLEAVED,
    "interleaved_bf16fp32_mmla_12x8",
    [](const GemmArgs &args) { return (args._Ksize>4); },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<interleaved_bf16fp32_mmla_12x8, bfloat16, float>(args); }
},
{ // gemm_bf16_interleaved
    GemmMethod::GEMM_INTERLEAVED,
    "interleaved_bf16fp32_dot_12x8",
    [](const GemmArgs &args) { return (args._Ksize>2); },
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<interleaved_bf16fp32_dot_12x8, bfloat16, float>(args); }
},
#endif // V8P6_BF
#ifdef __aarch64__
{
    GemmMethod::GEMM_INTERLEAVED,
    "sgemm_12x8",
    nullptr,
    nullptr,
    [](const GemmArgs &args) { return new GemmInterleaved<sgemm_12x8, bfloat16, float>(args); }
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
