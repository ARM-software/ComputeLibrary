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
#ifdef __aarch64__

#include "arm_gemm.hpp"
#include "gemm_common.hpp"
#include "gemm_implementation.hpp"
#include "gemm_interleaved.hpp"
#include "gemm_hybrid.hpp"
#include "gemm_native.hpp"

#include "kernels/a64_gemm_u16_12x8.hpp"
#include "kernels/a64_gemm_u8_12x8.hpp"
#include "kernels/a64_gemm_u8_4x4.hpp"
#include "kernels/a64_hybrid_u8u32_dot_16x4.hpp"
#include "kernels/sve_hybrid_u8u32_dot_4VLx4.hpp"
#include "kernels/sve_interleaved_u8u32_dot_3VLx8.hpp"
#include "kernels/sve_native_u8u32_dot_4VLx4.hpp"

namespace arm_gemm {

static const GemmImplementation<uint8_t, uint32_t> gemm_u8_methods[] = {
#ifdef __ARM_FEATURE_SVE
{
    GemmMethod::GEMM_HYBRID,
    "hybrid_u8u32_dot_4VLx4",
    [](const GemmArgs<uint32_t> &args) { return args._Ksize>=16 && args._alpha==1 && !args._trA && !args._trB && args._pretransposed_hint; },
    [](const GemmArgs<uint32_t> &args) { return ((args._Ksize <= 128) && (args._Nsize <= 128)) || ((args._nmulti > 1) && ((args._Msize / args._maxthreads) < 8)); },
    [](const GemmArgs<uint32_t> &args) { return new GemmHybrid<hybrid_u8u32_dot_4VLx4, uint8_t, uint32_t>(args); }
},
{
    GemmMethod::GEMM_NATIVE,
    "native_u8u32_dot_4VLx4",
    [](const GemmArgs<uint32_t> &args) { return (args._Ksize>=16 && args._alpha==1 && !args._trA && !args._trB); },
    [](const GemmArgs<uint32_t> &args) { return ((args._Ksize <= 128) && (args._Nsize <= 128)); },
    [](const GemmArgs<uint32_t> &args) { return new GemmNative<native_u8u32_dot_4VLx4, uint8_t, uint32_t>(args); }
},
{
    GemmMethod::GEMM_INTERLEAVED,
    "interleaved_u8u32_dot_3VLx8",
    [](const GemmArgs<uint32_t> &args) { return (args._Ksize>4); },
    nullptr,
    [](const GemmArgs<uint32_t> &args) { return new GemmInterleaved<interleaved_u8u32_dot_3VLx8, uint8_t, uint32_t>(args); }
},
#endif
{
    GemmMethod::GEMM_HYBRID,
    "hybrid_u8u32_dot_16x4",
    [](const GemmArgs<uint32_t> &args) { return args._ci->has_dotprod() && args._Ksize>=16 && !args._trA && !args._trB && args._pretransposed_hint; },
    [](const GemmArgs<uint32_t> &args) { return args._Nsize<=256 && args._Ksize>128; },
    [](const GemmArgs<uint32_t> &args) { return new GemmHybrid<hybrid_u8u32_dot_16x4, uint8_t, uint32_t>(args); }
},
{
    GemmMethod::GEMM_INTERLEAVED,
    "gemm_u8_12x8",
    [](const GemmArgs<uint32_t> &args) { return args._ci->has_dotprod(); },
    nullptr,
    [](const GemmArgs<uint32_t> &args) { return new GemmInterleaved<gemm_u8_12x8, uint8_t, uint32_t>(args); }
},
{
    GemmMethod::GEMM_INTERLEAVED,
    "gemm_u8_4x4",
    nullptr,
    nullptr,
    [](const GemmArgs<uint32_t> &args) { return new GemmInterleaved<gemm_u8_4x4, uint8_t, uint32_t>(args); }
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
const GemmImplementation<uint8_t, uint32_t> *gemm_implementation_list<uint8_t, uint32_t>() {
    return gemm_u8_methods;
}

/* Explicitly instantiate the external functions for these types. */
template UniqueGemmCommon<uint8_t, uint32_t> gemm<uint8_t, uint32_t>(const GemmArgs<uint32_t> &args);
template KernelDescription get_gemm_method<uint8_t, uint32_t>(const GemmArgs<uint32_t> &args);
template bool method_is_compatible<uint8_t, uint32_t>(GemmMethod method, const GemmArgs<uint32_t> &args);
template std::vector<KernelDescription> get_compatible_kernels<uint8_t, uint32_t> (const GemmArgs<uint32_t> &args);

} // namespace arm_gemm

#endif // __aarch64__
