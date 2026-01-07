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
#include "src/cpu/kernels/topkv/generic/neon/impl.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace cpu
{
namespace detail
{
static inline uint32_t reduce_u32x4(uint32x4_t v)
{
#if defined(__aarch64__)
    return vaddvq_u32(v);
#else
    uint32x2_t s = vadd_u32(vget_low_u32(v), vget_high_u32(v));
    s            = vpadd_u32(s, s);
    return vget_lane_u32(s, 0);
#endif
}

template <>
uint32_t count_gt_block<int32_t>(const int32_t *ptr, int32_t threshold)
{
    const int32x4_t  v   = vld1q_s32(ptr);
    const int32x4_t  thr = vdupq_n_s32(threshold);
    const uint32x4_t m   = vcgtq_s32(v, thr);  // 0xFFFFFFFF / 0 per lane
    const uint32x4_t b   = vshrq_n_u32(m, 31); // 0/1 per lane
    return reduce_u32x4(b);
}

} // namespace detail

void topkv_s32_neon(const ITensor *predictions, const ITensor *targets, ITensor *out, uint32_t k, const Window &win)
{
    detail::topkv_neon_wrapper<int32_t>(predictions, targets, out, k, win);
}

template void
detail::topkv_neon_wrapper<int32_t>(const ITensor *, const ITensor *, ITensor *, uint32_t, const Window &);

} // namespace cpu
} // namespace arm_compute
