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
static inline uint32_t reduce_u8_to_count(uint8x16_t m)
{
#if defined(__aarch64__)
    // mask is 0xFF where true, 0 otherwise -> shift to 0/1 then sum
    const uint8x16_t ones = vshrq_n_u8(m, 7);
    return vaddvq_u8(ones);
#else
    const uint8x16_t ones = vshrq_n_u8(m, 7);
    uint16x8_t       s16  = vpaddlq_u8(ones);
    uint32x4_t       s32  = vpaddlq_u16(s16);
    uint64x2_t       s64  = vpaddlq_u32(s32);
    return static_cast<uint32_t>(vgetq_lane_u64(s64, 0) + vgetq_lane_u64(s64, 1));
#endif
}

template <>
uint32_t count_gt_block<uint8_t>(const uint8_t *ptr, uint8_t threshold)
{
    const uint8x16_t v   = vld1q_u8(ptr);
    const uint8x16_t thr = vdupq_n_u8(threshold);
    const uint8x16_t m   = vcgtq_u8(v, thr); // 0xFF / 0x00 bytes
    return reduce_u8_to_count(m);
}

} // namespace detail

void topkv_qasymm8_neon(const ITensor *predictions, const ITensor *targets, ITensor *out, uint32_t k, const Window &win)
{
    detail::topkv_neon_wrapper<uint8_t>(predictions, targets, out, k, win);
}

template void
detail::topkv_neon_wrapper<uint8_t>(const ITensor *, const ITensor *, ITensor *, uint32_t, const Window &);

} // namespace cpu
} // namespace arm_compute
