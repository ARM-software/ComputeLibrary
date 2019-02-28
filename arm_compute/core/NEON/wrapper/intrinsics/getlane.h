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
#ifndef __ARM_COMPUTE_WRAPPER_GET_LANE_H__
#define __ARM_COMPUTE_WRAPPER_GET_LANE_H__

#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VGETLANE_IMPL_8(stype, vtype, postfix)                         \
    inline stype vgetlane(const vtype vector, const unsigned int lane) \
    {                                                                  \
        switch(lane)                                                   \
        {                                                              \
            case 0:                                                    \
                return vget_lane_##postfix(vector, 0);                 \
            case 1:                                                    \
                return vget_lane_##postfix(vector, 1);                 \
            case 2:                                                    \
                return vget_lane_##postfix(vector, 2);                 \
            case 3:                                                    \
                return vget_lane_##postfix(vector, 3);                 \
            case 4:                                                    \
                return vget_lane_##postfix(vector, 4);                 \
            case 5:                                                    \
                return vget_lane_##postfix(vector, 5);                 \
            case 6:                                                    \
                return vget_lane_##postfix(vector, 6);                 \
            case 7:                                                    \
                return vget_lane_##postfix(vector, 7);                 \
            default:                                                   \
                ARM_COMPUTE_ERROR("Invalid lane");                     \
        }                                                              \
    }

#define VGETLANE_IMPL_4(stype, vtype, postfix)                         \
    inline stype vgetlane(const vtype vector, const unsigned int lane) \
    {                                                                  \
        switch(lane)                                                   \
        {                                                              \
            case 0:                                                    \
                return vget_lane_##postfix(vector, 0);                 \
            case 1:                                                    \
                return vget_lane_##postfix(vector, 1);                 \
            case 2:                                                    \
                return vget_lane_##postfix(vector, 2);                 \
            case 3:                                                    \
                return vget_lane_##postfix(vector, 3);                 \
            default:                                                   \
                ARM_COMPUTE_ERROR("Invalid lane");                     \
        }                                                              \
    }

#define VGETLANE_IMPL_2(stype, vtype, postfix)                         \
    inline stype vgetlane(const vtype vector, const unsigned int lane) \
    {                                                                  \
        switch(lane)                                                   \
        {                                                              \
            case 0:                                                    \
                return vget_lane_##postfix(vector, 0);                 \
            case 1:                                                    \
                return vget_lane_##postfix(vector, 1);                 \
            default:                                                   \
                ARM_COMPUTE_ERROR("Invalid lane");                     \
        }                                                              \
    }

VGETLANE_IMPL_8(uint8_t, uint8x8_t, u8)
VGETLANE_IMPL_8(int8_t, int8x8_t, s8)
VGETLANE_IMPL_4(uint16_t, uint16x4_t, u16)
VGETLANE_IMPL_4(int16_t, int16x4_t, s16)
VGETLANE_IMPL_2(uint32_t, uint32x2_t, u32)
VGETLANE_IMPL_2(int32_t, int32x2_t, s32)
VGETLANE_IMPL_2(float, float32x2_t, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VGETLANE_IMPL_4(float16_t, float16x4_t, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#define VGETQLANE_IMPL_16(stype, vtype, postfix)                       \
    inline stype vgetlane(const vtype vector, const unsigned int lane) \
    {                                                                  \
        switch(lane)                                                   \
        {                                                              \
            case 0:                                                    \
                return vgetq_lane_##postfix(vector, 0);                \
            case 1:                                                    \
                return vgetq_lane_##postfix(vector, 1);                \
            case 2:                                                    \
                return vgetq_lane_##postfix(vector, 2);                \
            case 3:                                                    \
                return vgetq_lane_##postfix(vector, 3);                \
            case 4:                                                    \
                return vgetq_lane_##postfix(vector, 4);                \
            case 5:                                                    \
                return vgetq_lane_##postfix(vector, 5);                \
            case 6:                                                    \
                return vgetq_lane_##postfix(vector, 6);                \
            case 7:                                                    \
                return vgetq_lane_##postfix(vector, 7);                \
            case 8:                                                    \
                return vgetq_lane_##postfix(vector, 8);                \
            case 9:                                                    \
                return vgetq_lane_##postfix(vector, 9);                \
            case 10:                                                   \
                return vgetq_lane_##postfix(vector, 10);               \
            case 11:                                                   \
                return vgetq_lane_##postfix(vector, 11);               \
            case 12:                                                   \
                return vgetq_lane_##postfix(vector, 12);               \
            case 13:                                                   \
                return vgetq_lane_##postfix(vector, 13);               \
            case 14:                                                   \
                return vgetq_lane_##postfix(vector, 14);               \
            case 15:                                                   \
                return vgetq_lane_##postfix(vector, 15);               \
            default:                                                   \
                ARM_COMPUTE_ERROR("Invalid lane");                     \
        }                                                              \
    }

#define VGETQLANE_IMPL_8(stype, vtype, postfix)                        \
    inline stype vgetlane(const vtype vector, const unsigned int lane) \
    {                                                                  \
        switch(lane)                                                   \
        {                                                              \
            case 0:                                                    \
                return vgetq_lane_##postfix(vector, 0);                \
            case 1:                                                    \
                return vgetq_lane_##postfix(vector, 1);                \
            case 2:                                                    \
                return vgetq_lane_##postfix(vector, 2);                \
            case 3:                                                    \
                return vgetq_lane_##postfix(vector, 3);                \
            case 4:                                                    \
                return vgetq_lane_##postfix(vector, 4);                \
            case 5:                                                    \
                return vgetq_lane_##postfix(vector, 5);                \
            case 6:                                                    \
                return vgetq_lane_##postfix(vector, 6);                \
            case 7:                                                    \
                return vgetq_lane_##postfix(vector, 7);                \
            default:                                                   \
                ARM_COMPUTE_ERROR("Invalid lane");                     \
        }                                                              \
    }

#define VGETQLANE_IMPL_4(stype, vtype, postfix)                        \
    inline stype vgetlane(const vtype vector, const unsigned int lane) \
    {                                                                  \
        switch(lane)                                                   \
        {                                                              \
            case 0:                                                    \
                return vgetq_lane_##postfix(vector, 0);                \
            case 1:                                                    \
                return vgetq_lane_##postfix(vector, 1);                \
            case 2:                                                    \
                return vgetq_lane_##postfix(vector, 2);                \
            case 3:                                                    \
                return vgetq_lane_##postfix(vector, 3);                \
            default:                                                   \
                ARM_COMPUTE_ERROR("Invalid lane");                     \
        }                                                              \
    }

VGETQLANE_IMPL_16(uint8_t, uint8x16_t, u8)
VGETQLANE_IMPL_16(int8_t, int8x16_t, s8)
VGETQLANE_IMPL_8(uint16_t, uint16x8_t, u16)
VGETQLANE_IMPL_8(int16_t, int16x8_t, s16)
VGETQLANE_IMPL_4(uint32_t, uint32x4_t, u32)
VGETQLANE_IMPL_4(int32_t, int32x4_t, s32)
VGETQLANE_IMPL_4(float, float32x4_t, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VGETQLANE_IMPL_8(float16_t, float16x8_t, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VGETLANE_IMPL_8
#undef VGETLANE_IMPL_4
#undef VGETLANE_IMPL_2

#undef VGETQLANE_IMPL_16
#undef VGETQLANE_IMPL_8
#undef VGETQLANE_IMPL_4
} // namespace wrapper
} // namespace arm_compute
#endif /* __ARM_COMPUTE_WRAPPER_GET_LANE_H__ */
