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
#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"
#include "gemm_hybrid.hpp"
#include "gemm_hybrid_indirect.hpp"

#include "kernels/a64_gemm_u16_8x12.hpp"
#include "kernels/a64_gemm_u8_4x4.hpp"
#include "kernels/a64_gemm_u8_8x12.hpp"
#include "kernels/a64_hybrid_u8u32_dot_6x16.hpp"
#include "kernels/a64_hybrid_u8u32_mmla_6x16.hpp"
#include "kernels/a64_interleaved_u8u32_mmla_8x12.hpp"
#include "kernels/a64_smallK_hybrid_u8u32_dot_6x4.hpp"
#include "kernels/a64_smallK_hybrid_u8u32_dot_8x4.hpp"

#include "kernels/sve_hybrid_u8u32_dot_6x4VL.hpp"
#include "kernels/sve_hybrid_u8u32_mmla_6x4VL.hpp"
#include "kernels/sve_interleaved_u8u32_dot_8x3VL.hpp"
#include "kernels/sve_interleaved_u8u32_mmla_8x3VL.hpp"
#include "kernels/sve_smallK_hybrid_u8u32_dot_8x1VL.hpp"

namespace arm_gemm {

static const GemmImplementation<uint8_t, uint32_t> gemm_u8_methods[] = {
#ifdef ARM_COMPUTE_ENABLE_SVE
GemmImplementation<uint8_t, uint32_t>::with_estimate(
    GemmMethod::GEMM_HYBRID,
    "sve_hybrid_u8u32_mmla_6x4VL",
    [](const GemmArgs &args) { return args._ci->has_svei8mm(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_sve_hybrid_u8u32_mmla_6x4VL, uint8_t, uint32_t>::estimate_cycles<uint32_t>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_u8u32_mmla_6x4VL, uint8_t, uint32_t>(args); }
),
GemmImplementation<uint8_t, uint32_t>::with_estimate(
    GemmMethod::GEMM_INTERLEAVED,
    "sve_interleaved_u8u32_mmla_8x3VL",
    [](const GemmArgs &args) { return args._ci->has_svei8mm() && (args._Ksize>8); },
    [](const GemmArgs &args) { return GemmInterleaved<cls_sve_interleaved_u8u32_mmla_8x3VL, uint8_t, uint32_t>::estimate_cycles<uint32_t>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_u8u32_mmla_8x3VL, uint8_t, uint32_t>(args); }
),
{
    GemmMethod::GEMM_HYBRID,
    "sve_smallK_hybrid_u8u32_dot_8x1VL",
    [](const GemmArgs &args) { return args._ci->has_sve() && args._Ksize<=64 && !args._indirect_input; },
    [](const GemmArgs &args) { return !(args._ci->has_svei8mm() || args._ci->has_i8mm()); },
    [](const GemmArgs &args) { return new GemmHybrid<cls_sve_smallK_hybrid_u8u32_dot_8x1VL, uint8_t, uint32_t>(args); }
},
GemmImplementation<uint8_t, uint32_t>::with_estimate(
    GemmMethod::GEMM_HYBRID,
    "sve_hybrid_u8u32_dot_6x4VL",
    [](const GemmArgs &args) { return args._ci->has_sve(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_sve_hybrid_u8u32_dot_6x4VL, uint8_t, uint32_t>::estimate_cycles<uint32_t>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_sve_hybrid_u8u32_dot_6x4VL, uint8_t, uint32_t>(args); }
),
GemmImplementation<uint8_t, uint32_t>::with_estimate(
    GemmMethod::GEMM_INTERLEAVED,
    "sve_interleaved_u8u32_dot_8x3VL",
    [](const GemmArgs &args) { return args._ci->has_sve() && (args._Ksize>4); },
    [](const GemmArgs &args) { return GemmInterleaved<cls_sve_interleaved_u8u32_dot_8x3VL, uint8_t, uint32_t>::estimate_cycles<uint32_t>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_sve_interleaved_u8u32_dot_8x3VL, uint8_t, uint32_t>(args); }
),
#endif // ARM_COMPUTE_ENABLE_SVE
GemmImplementation<uint8_t, uint32_t>::with_estimate(
    GemmMethod::GEMM_INTERLEAVED,
    "a64_interleaved_u8u32_mmla_8x12",
    [](const GemmArgs &args) { return args._ci->has_i8mm() && (args._Ksize>8); },
    [](const GemmArgs &args) { return GemmInterleaved<cls_a64_interleaved_u8u32_mmla_8x12, uint8_t, uint32_t>::estimate_cycles<uint32_t>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_interleaved_u8u32_mmla_8x12, uint8_t, uint32_t>(args); }
),
GemmImplementation<uint8_t, uint32_t>::with_estimate(
    GemmMethod::GEMM_HYBRID,
    "a64_hybrid_u8u32_mmla_6x16",
    [](const GemmArgs &args) { return args._ci->has_i8mm(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_a64_hybrid_u8u32_mmla_6x16, uint8_t, uint32_t>::estimate_cycles<uint32_t>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_a64_hybrid_u8u32_mmla_6x16, uint8_t, uint32_t>(args); }
),
{
    GemmMethod::GEMM_HYBRID,
    "a64_smallK_hybrid_u8u32_dot_8x4",
    [](const GemmArgs &args) { return args._ci->has_dotprod() && (args._Nsize % 4 == 0) && (args._Ksize<=32) && !args._indirect_input; },
    [](const GemmArgs &args) { return !(args._ci->has_svei8mm() || args._ci->has_i8mm()); },
    [](const GemmArgs &args) { return new GemmHybrid<cls_a64_smallK_hybrid_u8u32_dot_8x4, uint8_t, uint32_t>(args); }
},
{
    GemmMethod::GEMM_HYBRID,
    "a64_smallK_hybrid_u8u32_dot_6x4",
    [](const GemmArgs &args) { return args._ci->has_dotprod() && (args._Nsize % 4 == 0) && (args._Ksize>32) && (args._Ksize<=64) && !args._indirect_input; },
    [](const GemmArgs &args) { return !(args._ci->has_svei8mm() || args._ci->has_i8mm()); },
    [](const GemmArgs &args) { return new GemmHybrid<cls_a64_smallK_hybrid_u8u32_dot_6x4, uint8_t, uint32_t>(args); }
},
{
    GemmMethod::GEMM_INTERLEAVED,
    "a64_gemm_u16_8x12",
    nullptr,
    [](const GemmArgs &args) { return args._ci->get_cpu_model() == CPUModel::A53 && args._Msize > 4; },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_gemm_u16_8x12, uint8_t, uint32_t>(args); },
},
GemmImplementation<uint8_t, uint32_t>::with_estimate(
    GemmMethod::GEMM_HYBRID,
    "a64_hybrid_u8u32_dot_6x16",
    [](const GemmArgs &args) { return args._ci->has_dotprod(); },
    [](const GemmArgs &args) { return GemmHybridIndirect<cls_a64_hybrid_u8u32_dot_6x16, uint8_t, uint32_t>::estimate_cycles<uint32_t>(args); },
    [](const GemmArgs &args) { return new GemmHybridIndirect<cls_a64_hybrid_u8u32_dot_6x16, uint8_t, uint32_t>(args); }
),
GemmImplementation<uint8_t, uint32_t>::with_estimate(
    GemmMethod::GEMM_INTERLEAVED,
    "a64_gemm_u8_8x12",
    [](const GemmArgs &args) { return args._ci->has_dotprod(); },
    [](const GemmArgs &args) { return GemmInterleaved<cls_a64_gemm_u8_8x12, uint8_t, uint32_t>::estimate_cycles<uint32_t>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_gemm_u8_8x12, uint8_t, uint32_t>(args); }
),
GemmImplementation<uint8_t, uint32_t>::with_estimate(
    GemmMethod::GEMM_INTERLEAVED,
    "a64_gemm_u8_4x4",
    nullptr,
    [](const GemmArgs &args) { return GemmInterleaved<cls_a64_gemm_u8_4x4, uint8_t, uint32_t>::estimate_cycles<uint32_t>(args); },
    [](const GemmArgs &args) { return new GemmInterleaved<cls_a64_gemm_u8_4x4, uint8_t, uint32_t>(args); }
),
{
    GemmMethod::DEFAULT,
    "",
    nullptr,
    nullptr,
    nullptr
}
};

template<>
const GemmImplementation<uint8_t, uint32_t> *gemm_implementation_list<uint8_t, uint32_t>() {
    return gemm_u8_methods;
}

/* Explicitly instantiate the external functions for these types. */
template UniqueGemmCommon<uint8_t, uint32_t> gemm<uint8_t, uint32_t, Nothing>(const GemmArgs &args, const Nothing &);
template KernelDescription get_gemm_method<uint8_t, uint32_t, Nothing>(const GemmArgs &args, const Nothing &);
template std::vector<KernelDescription> get_compatible_kernels<uint8_t, uint32_t, Nothing> (const GemmArgs &args, const Nothing &);

} // namespace arm_gemm

#endif // __aarch64__
