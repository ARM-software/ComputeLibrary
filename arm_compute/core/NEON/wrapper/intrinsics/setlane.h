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
#ifndef __ARM_COMPUTE_WRAPPER_SET_LANE_H__
#define __ARM_COMPUTE_WRAPPER_SET_LANE_H__

#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VSETLANE_IMPL_8(stype, atype, vtype, postfix)                                     \
    inline stype vsetlane(const atype value, const vtype vector, const unsigned int lane) \
    {                                                                                     \
        switch(lane)                                                                      \
        {                                                                                 \
            case 0:                                                                       \
                return vset_lane_##postfix(value, vector, 0);                             \
            case 1:                                                                       \
                return vset_lane_##postfix(value, vector, 1);                             \
            case 2:                                                                       \
                return vset_lane_##postfix(value, vector, 2);                             \
            case 3:                                                                       \
                return vset_lane_##postfix(value, vector, 3);                             \
            case 4:                                                                       \
                return vset_lane_##postfix(value, vector, 4);                             \
            case 5:                                                                       \
                return vset_lane_##postfix(value, vector, 5);                             \
            case 6:                                                                       \
                return vset_lane_##postfix(value, vector, 6);                             \
            case 7:                                                                       \
                return vset_lane_##postfix(value, vector, 7);                             \
            default:                                                                      \
                ARM_COMPUTE_ERROR("Invalid lane");                                        \
        }                                                                                 \
    }

#define VSETLANE_IMPL_4(stype, atype, vtype, postfix)                                     \
    inline stype vsetlane(const atype value, const vtype vector, const unsigned int lane) \
    {                                                                                     \
        switch(lane)                                                                      \
        {                                                                                 \
            case 0:                                                                       \
                return vset_lane_##postfix(value, vector, 0);                             \
            case 1:                                                                       \
                return vset_lane_##postfix(value, vector, 1);                             \
            case 2:                                                                       \
                return vset_lane_##postfix(value, vector, 2);                             \
            case 3:                                                                       \
                return vset_lane_##postfix(value, vector, 3);                             \
            default:                                                                      \
                ARM_COMPUTE_ERROR("Invalid lane");                                        \
        }                                                                                 \
    }

#define VSETLANE_IMPL_2(stype, atype, vtype, postfix)                                     \
    inline stype vsetlane(const atype value, const vtype vector, const unsigned int lane) \
    {                                                                                     \
        switch(lane)                                                                      \
        {                                                                                 \
            case 0:                                                                       \
                return vset_lane_##postfix(value, vector, 0);                             \
            case 1:                                                                       \
                return vset_lane_##postfix(value, vector, 1);                             \
            default:                                                                      \
                ARM_COMPUTE_ERROR("Invalid lane");                                        \
        }                                                                                 \
    }

VSETLANE_IMPL_8(uint8x8_t, uint8_t, uint8x8_t, u8)
VSETLANE_IMPL_8(int8x8_t, int8_t, int8x8_t, s8)
VSETLANE_IMPL_4(uint16x4_t, uint16_t, uint16x4_t, u16)
VSETLANE_IMPL_4(int16x4_t, int16_t, int16x4_t, s16)
VSETLANE_IMPL_2(uint32x2_t, uint32_t, uint32x2_t, u32)
VSETLANE_IMPL_2(int32x2_t, int32_t, int32x2_t, s32)
VSETLANE_IMPL_2(float32x2_t, float, float32x2_t, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VSETLANE_IMPL_4(float16x4_t, float16_t, float16x4_t, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#define VSETQLANE_IMPL_16(stype, atype, vtype, postfix)                                   \
    inline stype vsetlane(const atype value, const vtype vector, const unsigned int lane) \
    {                                                                                     \
        switch(lane)                                                                      \
        {                                                                                 \
            case 0:                                                                       \
                return vsetq_lane_##postfix(value, vector, 0);                            \
            case 1:                                                                       \
                return vsetq_lane_##postfix(value, vector, 1);                            \
            case 2:                                                                       \
                return vsetq_lane_##postfix(value, vector, 2);                            \
            case 3:                                                                       \
                return vsetq_lane_##postfix(value, vector, 3);                            \
            case 4:                                                                       \
                return vsetq_lane_##postfix(value, vector, 4);                            \
            case 5:                                                                       \
                return vsetq_lane_##postfix(value, vector, 5);                            \
            case 6:                                                                       \
                return vsetq_lane_##postfix(value, vector, 6);                            \
            case 7:                                                                       \
                return vsetq_lane_##postfix(value, vector, 7);                            \
            case 8:                                                                       \
                return vsetq_lane_##postfix(value, vector, 8);                            \
            case 9:                                                                       \
                return vsetq_lane_##postfix(value, vector, 9);                            \
            case 10:                                                                      \
                return vsetq_lane_##postfix(value, vector, 10);                           \
            case 11:                                                                      \
                return vsetq_lane_##postfix(value, vector, 11);                           \
            case 12:                                                                      \
                return vsetq_lane_##postfix(value, vector, 12);                           \
            case 13:                                                                      \
                return vsetq_lane_##postfix(value, vector, 13);                           \
            case 14:                                                                      \
                return vsetq_lane_##postfix(value, vector, 14);                           \
            case 15:                                                                      \
                return vsetq_lane_##postfix(value, vector, 15);                           \
            default:                                                                      \
                ARM_COMPUTE_ERROR("Invalid lane");                                        \
        }                                                                                 \
    }

#define VSETQLANE_IMPL_8(stype, atype, vtype, postfix)                                    \
    inline stype vsetlane(const atype value, const vtype vector, const unsigned int lane) \
    {                                                                                     \
        switch(lane)                                                                      \
        {                                                                                 \
            case 0:                                                                       \
                return vsetq_lane_##postfix(value, vector, 0);                            \
            case 1:                                                                       \
                return vsetq_lane_##postfix(value, vector, 1);                            \
            case 2:                                                                       \
                return vsetq_lane_##postfix(value, vector, 2);                            \
            case 3:                                                                       \
                return vsetq_lane_##postfix(value, vector, 3);                            \
            case 4:                                                                       \
                return vsetq_lane_##postfix(value, vector, 4);                            \
            case 5:                                                                       \
                return vsetq_lane_##postfix(value, vector, 5);                            \
            case 6:                                                                       \
                return vsetq_lane_##postfix(value, vector, 6);                            \
            case 7:                                                                       \
                return vsetq_lane_##postfix(value, vector, 7);                            \
            default:                                                                      \
                ARM_COMPUTE_ERROR("Invalid lane");                                        \
        }                                                                                 \
    }

#define VSETQLANE_IMPL_4(stype, atype, vtype, postfix)                                    \
    inline stype vsetlane(const atype value, const vtype vector, const unsigned int lane) \
    {                                                                                     \
        switch(lane)                                                                      \
        {                                                                                 \
            case 0:                                                                       \
                return vsetq_lane_##postfix(value, vector, 0);                            \
            case 1:                                                                       \
                return vsetq_lane_##postfix(value, vector, 1);                            \
            case 2:                                                                       \
                return vsetq_lane_##postfix(value, vector, 2);                            \
            case 3:                                                                       \
                return vsetq_lane_##postfix(value, vector, 3);                            \
            default:                                                                      \
                ARM_COMPUTE_ERROR("Invalid lane");                                        \
        }                                                                                 \
    }

VSETQLANE_IMPL_16(uint8x16_t, uint8_t, uint8x16_t, u8)
VSETQLANE_IMPL_16(int8x16_t, int8_t, int8x16_t, s8)
VSETQLANE_IMPL_8(uint16x8_t, uint16_t, uint16x8_t, u16)
VSETQLANE_IMPL_8(int16x8_t, int16_t, int16x8_t, s16)
VSETQLANE_IMPL_4(uint32x4_t, uint32_t, uint32x4_t, u32)
VSETQLANE_IMPL_4(int32x4_t, int32_t, int32x4_t, s32)
VSETQLANE_IMPL_4(float32x4_t, float, float32x4_t, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VSETQLANE_IMPL_8(float16x8_t, float16_t, float16x8_t, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef VSETLANE_IMPL_8
#undef VSETLANE_IMPL_4
#undef VSETLANE_IMPL_2

#undef VSETQLANE_IMPL_16
#undef VSETQLANE_IMPL_8
#undef VSETQLANE_IMPL_4
} // namespace wrapper
} // namespace arm_compute
#endif /* __ARM_COMPUTE_WRAPPER_AET_LANE_H__ */
