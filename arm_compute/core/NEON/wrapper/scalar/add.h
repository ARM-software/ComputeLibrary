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
#ifndef __ARM_COMPUTE_WRAPPER_SCALAR_ADD_H__
#define __ARM_COMPUTE_WRAPPER_SCALAR_ADD_H__

#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
inline uint8_t add_sat(const uint8_t &a, const uint8_t &b)
{
    const uint8x8_t va = { a, 0, 0, 0, 0, 0, 0, 0 };
    const uint8x8_t vb = { b, 0, 0, 0, 0, 0, 0, 0 };
    return vget_lane_u8(vqadd_u8(va, vb), 0);
}

inline int16_t add_sat(const int16_t &a, const int16_t &b)
{
    const int16x4_t va = { a, 0, 0, 0 };
    const int16x4_t vb = { b, 0, 0, 0 };
    return vget_lane_s16(vqadd_s16(va, vb), 0);
}

inline float add_sat(const float &a, const float &b)
{
    // No notion of saturation exists in floating point
    return a + b;
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
inline float16_t add_sat(const float16_t &a, const float16_t &b)
{
    // No notion of saturation exists in floating point
    return a + b;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
} // namespace wrapper
} // namespace arm_compute
#endif /* __ARM_COMPUTE_WRAPPER_SCALAR_ADD_H__ */
