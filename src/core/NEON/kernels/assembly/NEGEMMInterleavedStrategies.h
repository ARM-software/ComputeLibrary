/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEGEMMINTERLEAVEDSTRATEGIES_H__
#define __ARM_COMPUTE_NEGEMMINTERLEAVEDSTRATEGIES_H__

#include "../arm_gemm/utils.hpp"
#include "arm_gemm.hpp"

#include "../arm_gemm/mergeresults.hpp"
#include "../arm_gemm/transform.hpp"

#include "../arm_gemm/kernels/a32_sgemm_8x6.hpp"
#include "../arm_gemm/kernels/a64_gemm_s8_12x8.hpp"
#include "../arm_gemm/kernels/a64_gemm_s8_4x4.hpp"
#include "../arm_gemm/kernels/a64_gemm_u8_12x8.hpp"
#include "../arm_gemm/kernels/a64_gemm_u8_4x4.hpp"
#include "../arm_gemm/kernels/a64_hgemm_24x8.hpp"
#include "../arm_gemm/kernels/a64_sgemm_12x8.hpp"

namespace arm_compute
{
namespace
{
template <typename To, bool use_dot = false>
struct Kernel
{
};

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <>
struct Kernel<float16_t, false>
{
    using strategy = arm_gemm::hgemm_24x8;
};
#endif /*__ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
#ifdef __aarch64__
template <>
struct Kernel<float, false>
{
    using strategy = arm_gemm::sgemm_12x8;
};
template <>
struct Kernel<int8_t, false>
{
    using strategy = arm_gemm::gemm_s8_4x4;
};
template <>
struct Kernel<uint8_t, false>
{
    using strategy = arm_gemm::gemm_u8_4x4;
};

//Use different strategies for 8bit dot product:
template <>
struct Kernel<int8_t, true>
{
    using strategy = arm_gemm::gemm_s8_12x8;
};
template <>
struct Kernel<uint8_t, true>
{
    using strategy = arm_gemm::gemm_u8_12x8;
};
#else
template <>
struct Kernel<float, false>
{
    using strategy = arm_gemm::sgemm_8x6;
};
#endif /* __aarch64__ */

} // namespace
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEGEMMINTERLEAVEDSTRATEGIES_H__ */
