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
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)

#include "src/core/NEON/wrapper/wrapper.h"
#include "src/cpu/kernels/topkv/generic/neon/impl.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace cpu
{
namespace detail
{
static inline uint32_t reduce_u16x8(uint16x8_t v)
{
#if defined(__aarch64__)
    return vaddvq_u16(v);
#else
    uint16x4_t s = vadd_u16(vget_low_u16(v), vget_high_u16(v));
    s            = vpadd_u16(s, s);
    return vget_lane_u16(s, 0);
#endif
}

template <>
uint32_t count_gt_block<float16_t>(const float16_t *ptr, float16x8_t thr_vec)
{
    const float16x8_t v    = wrapper::vloadq(ptr);
    const uint16x8_t  mask = wrapper::vcgt(v, thr_vec); // new: v > (threshold)
    const uint16x8_t  b    = wrapper::vshrq_n<15>(mask);
    return reduce_u16x8(b);
}

} // namespace detail

void topkv_fp16_neon(const ITensor *predictions, const ITensor *targets, ITensor *out, uint32_t k, const Window &win)
{
    detail::topkv_neon_wrapper<float16_t>(predictions, targets, out, k, win);
}

template void
detail::topkv_neon_wrapper<float16_t>(const ITensor *, const ITensor *, ITensor *, uint32_t, const Window &);

} // namespace cpu
} // namespace arm_compute

#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS) */
