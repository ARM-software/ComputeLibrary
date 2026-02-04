/*
 * Copyright (c) 2026 Arm Limited.
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
#if defined(__ARM_FEATURE_SVE)

#include "src/cpu/kernels/topkv/generic/sve/impl.h"

#include <arm_sve.h>
#include <cstdint>

namespace arm_compute
{
namespace cpu
{
namespace detail
{

template <>
inline uint32_t vector_length<int8_t>()
{
    return static_cast<uint32_t>(svcntb());
}

template <>
inline uint32_t count_gt_block<int8_t>(const int8_t *ptr, int8_t thr, uint32_t block_elems)
{
    const svbool_t pg = svwhilelt_b8(static_cast<uint64_t>(0), static_cast<uint64_t>(block_elems));
    const svint8_t v  = svld1_s8(pg, ptr);
    const svbool_t gt = svcmpgt_n_s8(pg, v, thr);
    return static_cast<uint32_t>(svcntp_b8(svptrue_b8(), gt));
}

} // namespace detail

void topkv_qasymm8_signed_sve(
    const ITensor *predictions, const ITensor *targets, ITensor *out, uint32_t k, const Window &win)
{
    detail::topkv_sve_wrapper<int8_t>(predictions, targets, out, k, win);
}

// Force instantiation into this TU
template void detail::topkv_sve_wrapper<int8_t>(const ITensor *, const ITensor *, ITensor *, uint32_t, const Window &);

} // namespace cpu
} // namespace arm_compute

#endif // __ARM_FEATURE_SVE
