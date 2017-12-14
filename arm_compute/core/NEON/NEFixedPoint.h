/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEFIXEDPOINT_H__
#define __ARM_COMPUTE_NEFIXEDPOINT_H__

#include "arm_compute/core/FixedPoint.h"

#include <arm_neon.h>

namespace arm_compute
{
using qint8x8_t    = int8x8_t;    /**< 8 bit fixed point vector with 8 elements */
using qint8x8x2_t  = int8x8x2_t;  /**< 8 bit fixed point vector with 16 elements */
using qint8x8x3_t  = int8x8x3_t;  /**< 8 bit fixed point vector with 24 elements */
using qint8x8x4_t  = int8x8x4_t;  /**< 8 bit fixed point vector with 32 elements */
using qint8x16_t   = int8x16_t;   /**< 8 bit fixed point vector with 16 elements */
using qint8x16x2_t = int8x16x2_t; /**< 8 bit fixed point vector with 32 elements */
using qint8x16x3_t = int8x16x3_t; /**< 8 bit fixed point vector with 48 elements */
using qint8x16x4_t = int8x16x4_t; /**< 8 bit fixed point vector with 64 elements */
using qint16x4_t   = int16x4_t;   /**< 16 bit fixed point vector with 4 elements */
using qint16x4x2_t = int16x4x2_t; /**< 16 bit fixed point vector with 8 elements */
using qint16x4x3_t = int16x4x3_t; /**< 16 bit fixed point vector with 12 elements */
using qint16x4x4_t = int16x4x4_t; /**< 16 bit fixed point vector with 16 elements */
using qint16x8_t   = int16x8_t;   /**< 16 bit fixed point vector with 8 elements */
using qint16x8x2_t = int16x8x2_t; /**< 16 bit fixed point vector with 16 elements */
using qint16x8x3_t = int16x8x3_t; /**< 16 bit fixed point vector with 24 elements */
using qint16x8x4_t = int16x8x4_t; /**< 16 bit fixed point vector with 32 elements */
using qint32x2_t   = int32x2_t;   /**< 32 bit fixed point vector with 2 elements */
using qint32x4_t   = int32x4_t;   /**< 32 bit fixed point vector with 4 elements */
using qint32x4x2_t = int32x4x2_t; /**< 32 bit fixed point vector with 8 elements */

/** Get the lower half of a 16 elements vector
 *
 * @param[in] a vector of 16 elements
 *
 * @return 8 bit fixed point vector (8 elements)
 */
qint8x8_t vget_low_qs8(qint8x16_t a);

/** Get the lower half of a 16 elements vector
 *
 * @param[in] a vector of 8 elements
 *
 * @return 16 bit fixed point vector (4 elements)
 */
qint16x4_t vget_low_qs16(qint16x8_t a);

/** Get the higher half of a 16 elements vector
 *
 * @param[in] a vector of 16 elements
 *
 * @return 8 bit fixed point vector (8 elements)
 */
qint8x8_t vget_high_qs8(qint8x16_t a);

/** Get the higher half of a 16 elements vector
 *
 * @param[in] a vector of 8 elements
 *
 * @return 16 bit fixed point vector (4 elements)
 */
qint16x4_t vget_high_qs16(qint16x8_t a);

/** Load a single 8 bit fixed point vector from memory (8 elements)
 *
 * @param[in] addr Memory address of the 8 bit fixed point vector to load
 *
 * @return 8 bit fixed point vector (8 elements)
 */
qint8x8_t vld1_qs8(const qint8_t *addr);

/** Load a single 16 bit fixed point vector from memory (4 elements)
 *
 * @param[in] addr Memory address of the 16 bit fixed point vector to load
 *
 * @return 16 bit fixed point vector (4 elements)
 */
qint16x4_t vld1_qs16(const qint16_t *addr);

/** Load a single 8 bit fixed point vector from memory (16 elements)
 *
 * @param[in] addr Memory address of the 8 bit fixed point vector to load
 *
 * @return 8 bit fixed point vector (16 elements)
 */
qint8x16_t vld1q_qs8(const qint8_t *addr);

/** Load a single 16 bit fixed point vector from memory (8 elements)
 *
 * @param[in] addr Memory address of the 16 bit fixed point vector to load
 *
 * @return 16 bit fixed point vector (8 elements)
 */
qint16x8_t vld1q_qs16(const qint16_t *addr);

/** Load all lanes of 8 bit fixed point vector with same value from memory (8 elements)
 *
 * @param[in] addr Memory address of the 8 bit fixed point scalar value to load
 *
 * @return 8 bit fixed point vector (8 elements)
 */
qint8x8_t vld1_dup_qs8(const qint8_t *addr);

/** Load all lanes of 16 bit fixed point vector with same value from memory (4 elements)
 *
 * @param[in] addr Memory address of the 16 bit fixed point scalar value to load
 *
 * @return 16 bit fixed point vector (4 elements)
 */
qint16x4_t vld1_dup_qs16(const qint16_t *addr);

/** Load all lanes of 8 bit fixed point vector with same value from memory (16 elements)
 *
 * @param[in] addr Memory address of the 8 bit fixed point scalar value to load
 *
 * @return 8 bit fixed point vector (16 elements)
 */
qint8x16_t vld1q_dup_qs8(const qint8_t *addr);

/** Load all lanes of 16 bit fixed point vector with same value from memory (8 elements)
 *
 * @param[in] addr Memory address of the 16 bit fixed point scalar value to load
 *
 * @return 16 bit fixed point vector (8 elements)
 */
qint16x8_t vld1q_dup_qs16(const qint16_t *addr);

/** Load two 16 bit fixed point vectors from memory (8x2 elements)
 *
 * @param[in] addr Memory address of the 16 bit fixed point vectors to load
 *
 * @return 16 bit fixed point vectors (8x2 elements)
 */
qint16x8x2_t vld2q_qs16(qint16_t *addr);

/** Store a single 8 bit fixed point vector to memory (8 elements)
 *
 * @param[in] addr Memory address where the 8 bit fixed point vector should be stored
 * @param[in] b    8 bit fixed point vector to store
 *
 */
void vst1_qs8(qint8_t *addr, qint8x8_t b);

/** Store a single 16 bit fixed point vector to memory (4 elements)
 *
 * @param[in] addr Memory address where the 16 bit fixed point vector should be stored
 * @param[in] b    16 bit fixed point vector to store
 *
 */
void vst1_qs16(qint16_t *addr, qint16x4_t b);

/** Store a single 8 bit fixed point vector to memory (16 elements)
 *
 * @param[in] addr Memory address where the 8 bit fixed point vector should be stored
 * @param[in] b    8 bit fixed point vector to store
 *
 */
void vst1q_qs8(qint8_t *addr, qint8x16_t b);

/** Store a single 16 bit fixed point vector to memory (8 elements)
 *
 * @param[in] addr Memory address where the 16 bit fixed point vector should be stored
 * @param[in] b    16 bit fixed point vector to store
 *
 */
void vst1q_qs16(qint16_t *addr, qint16x8_t b);

/** Store two 16 bit fixed point vector to memory (8x2 elements)
 *
 * @param[in] addr Memory address where the 16 bit fixed point vectors should be stored
 * @param[in] b    16 bit fixed point vectors to store
 *
 */
void vst2q_qs16(qint16_t *addr, qint16x8x2_t b);

/** 16 bit fixed point vector saturating narrow (8 elements)
 *
 * @param[in] a 16 bit fixed point vector to convert
 *
 * @return 8 bit fixed point vector
 */
qint8x8_t vqmovn_q16(qint16x8_t a);

/** 32 bit fixed point vector saturating narrow (4 elements)
 *
 * @param[in] a 32 bit fixed point vector to convert
 *
 * @return 16 bit fixed point vector
 */
qint16x4_t vqmovn_q32(qint32x4_t a);

/** 8 bit fixed point vector duplicate (8 elements)
 *
 * @param[in] a 8 bit fixed point to duplicate
 *
 * @return The result of the vector duplication
 */
qint8x8_t vdup_n_qs8(qint8_t a);

/** 16 bit fixed point vector duplicate (4 elements)
 *
 * @param[in] a 16 bit fixed point to duplicate
 *
 * @return The result of the vector duplication
 */
qint16x4_t vdup_n_qs16(qint16_t a);

/** 8 bit fixed point vector duplicate (16 elements)
 *
 * @param[in] a 8 bit fixed point to duplicate
 *
 * @return The result of the vector duplication
 */
qint8x16_t vdupq_n_qs8(qint8_t a);

/** Duplicate a float and convert it to 8 bit fixed point vector (16 elements)
 *
 * @param[in] a                    floating point value to convert and duplicate
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the vector duplication
 */
qint8x16_t vdupq_n_qs8_f32(float a, int fixed_point_position);

/** Duplicate a float and convert it to 16 bit fixed point vector (8 elements)
 *
 * @param[in] a                    floating point value to convert and duplicate
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the vector duplication
 */
qint16x8_t vdupq_n_qs16_f32(float a, int fixed_point_position);

/** 16 bit fixed point vector duplicate (8 elements)
 *
 * @param[in] a 16 bit fixed point to duplicate
 *
 * @return The result of the vector duplication
 */
qint16x8_t vdupq_n_qs16(qint16x8_t a);

/** Absolute value of 8 bit fixed point vector (8 elements)
 *
 * @param[in] a 8 bit fixed point input vector
 *
 * @return The result of the 8 bit fixed point vector absolute value
 */
qint8x8_t vabs_qs8(qint8x8_t a);

/** Absolute value of 16 bit fixed point vector (4 elements)
 *
 * @param[in] a 16 bit fixed point input vector
 *
 * @return The result of the 16 bit fixed point vector absolute value
 */
qint16x4_t vabs_qs16(qint16x4_t a);

/** Absolute value of 8 bit fixed point vector (16 elements)
 *
 * @param[in] a 8 bit fixed point input vector
 *
 * @return The result of the 8 bit fixed point vector absolute value
 */
qint8x16_t vabsq_qs8(qint8x16_t a);

/** Absolute value of 16 bit fixed point vector (8 elements)
 *
 * @param[in] a 16 bit fixed point input vector
 *
 * @return The result of the 16 bit fixed point vector absolute value
 */
qint16x8_t vabsq_qs16(qint16x8_t a);

/** Saturating absolute value of 8 bit fixed point vector (8 elements)
 *
 * @param[in] a 8 bit fixed point input vector
 *
 * @return The result of the 8 bit fixed point vector absolute value
 */
qint8x8_t vqabs_qs8(qint8x8_t a);

/** Saturating absolute value of 16 bit fixed point vector (4 elements)
 *
 * @param[in] a 4 bit fixed point input vector
 *
 * @return The result of the 16 bit fixed point vector absolute value
 */
qint16x4_t vqabs_qs16(qint16x4_t a);

/** Saturating absolute value of 8 bit fixed point vector (16 elements)
 *
 * @param[in] a 8 bit fixed point input vector
 *
 * @return The result of the 8 bit fixed point vector absolute value
 */
qint8x16_t vqabsq_qs8(qint8x16_t a);

/** Saturating absolute value of 16 bit fixed point vector (8 elements)
 *
 * @param[in] a 16 bit fixed point input vector
 *
 * @return The result of the 16 bit fixed point vector absolute value
 */
qint16x8_t vqabsq_qs16(qint16x8_t a);

/** 8 bit fixed point vector max (8 elements)
 *
 * @param[in] a First 8 bit fixed point input vector
 * @param[in] b Second 8 bit fixed point input vector
 *
 * @return The result of the 8 bit fixed point vector max operation
 */
qint8x8_t vmax_qs8(qint8x8_t a, qint8x8_t b);

/** 16 bit fixed point vector max (4 elements)
 *
 * @param[in] a First 16 bit fixed point input vector
 * @param[in] b Second 16 bit fixed point input vector
 *
 * @return The result of the 16 bit fixed point vector max operation
 */
qint16x4_t vmax_qs16(qint16x4_t a, qint16x4_t b);

/** 8 bit fixed point vector max (16 elements)
 *
 * @param[in] a First 8 bit fixed point input vector
 * @param[in] b Second 8 bit fixed point input vector
 *
 * @return The result of the 8 bit fixed point vector max operation
 */
qint8x16_t vmaxq_qs8(qint8x16_t a, qint8x16_t b);

/** 16 bit fixed point vector max (8 elements)
 *
 * @param[in] a First 16 bit fixed point input vector
 * @param[in] b Second 16 bit fixed point input vector
 *
 * @return The result of the 16 bit fixed point vector max operation
 */
qint16x8_t vmaxq_qs16(qint16x8_t a, qint16x8_t b);

/** 8 bit fixed point vector pairwise max (8 elements)
 *
 * @param[in] a First 8 bit fixed point input vector
 * @param[in] b Second 8 bit fixed point input vector
 *
 * @return The result of the 8 bit fixed point vector pairwise max operation
 */
qint8x8_t vpmax_qs8(qint8x8_t a, qint8x8_t b);

/** 16 bit fixed point vector pairwise max (4 elements)
 *
 * @param[in] a First 16 bit fixed point input vector
 * @param[in] b Second 16 bit fixed point input vector
 *
 * @return The result of the 16 bit fixed point vector pairwise max operation
 */
qint16x4_t vpmax_qs16(qint16x4_t a, qint16x4_t b);

/** 8 bit fixed point vector min (8 elements)
 *
 * @param[in] a First 8 bit fixed point input vector
 * @param[in] b Second 8 bit fixed point input vector
 *
 * @return The result of the 8 bit fixed point vector max operation
 */
qint8x8_t vmin_qs8(qint8x8_t a, qint8x8_t b);

/** 16 bit fixed point vector min (4 elements)
 *
 * @param[in] a First 16 bit fixed point input vector
 * @param[in] b Second 16 bit fixed point input vector
 *
 * @return The result of the 16 bit fixed point vector max operation
 */
qint16x4_t vmin_qs16(qint16x4_t a, qint16x4_t b);

/** 8 bit fixed point vector min (16 elements)
 *
 * @param[in] a First 8 bit fixed point input vector
 * @param[in] b Second 8 bit fixed point input vector
 *
 * @return The result of the 8 bit fixed point vector min operation
 */
qint8x16_t vminq_qs8(qint8x16_t a, qint8x16_t b);

/** 16 bit fixed point vector min (8 elements)
 *
 * @param[in] a First 16 bit fixed point input vector
 * @param[in] b Second 16 bit fixed point input vector
 *
 * @return The result of the 16 bit fixed point vector min operation
 */
qint16x8_t vminq_qs16(qint16x8_t a, qint16x8_t b);

/** 8 bit fixed point vector pairwise min (8 elements)
 *
 * @param[in] a First 8 bit fixed point input vector
 * @param[in] b Second 8 bit fixed point input vector
 *
 * @return The result of the 8 bit fixed point vector pairwise min operation
 */
qint8x8_t vpmin_qs8(qint8x8_t a, qint8x8_t b);

/** 16 bit fixed point vector pairwise min (4 elements)
 *
 * @param[in] a First 16 bit fixed point input vector
 * @param[in] b Second 16 bit fixed point input vector
 *
 * @return The result of the 16 bit fixed point vector pairwise min operation
 */
qint16x4_t vpmin_qs16(qint16x4_t a, qint16x4_t b);

/** 8 bit fixed point vector add (8 elements)
 *
 * @param[in] a First 8 bit fixed point input vector
 * @param[in] b Second 8 bit fixed point input vector
 *
 * @return The result of the 8 bit fixed point vector addition
 */
qint8x8_t vadd_qs8(qint8x8_t a, qint8x8_t b);

/** 16 bit fixed point vector add (4 elements)
 *
 * @param[in] a First 16 bit fixed point input vector
 * @param[in] b Second 16 bit fixed point input vector
 *
 * @return The result of the 16 bit fixed point vector addition
 */
qint16x4_t vadd_qs16(qint16x4_t a, qint16x4_t b);

/** 8 bit fixed point vector add (16 elements)
 *
 * @param[in] a First 8 bit fixed point input vector
 * @param[in] b Second 8 bit fixed point input vector
 *
 * @return The result of the 8 bit fixed point vector addition
 */
qint8x16_t vaddq_qs8(qint8x16_t a, qint8x16_t b);

/** 16 bit fixed point vector add (8 elements)
 *
 * @param[in] a First 16 bit fixed point input vector
 * @param[in] b Second 16 bit fixed point input vector
 *
 * @return The result of the 16 bit fixed point vector addition
 */
qint16x8_t vaddq_qs16(qint16x8_t a, qint16x8_t b);

/** 8 bit fixed point vector saturating add (8 elements)
 *
 * @param[in] a First 8 bit fixed point input vector
 * @param[in] b Second 8 bit fixed point input vector
 *
 * @return The result of the 8 bit fixed point vector addition. The result is saturated in case of overflow
 */
qint8x8_t vqadd_qs8(qint8x8_t a, qint8x8_t b);

/** 16 bit fixed point vector saturating add (4 elements)
 *
 * @param[in] a First 16 bit fixed point input vector
 * @param[in] b Second 16 bit fixed point input vector
 *
 * @return The result of the 16 bit fixed point vector addition. The result is saturated in case of overflow
 */
qint16x4_t vqadd_qs16(qint16x4_t a, qint16x4_t b);

/** 8 bit fixed point vector saturating add (16 elements)
 *
 * @param[in] a First 8 bit fixed point input vector
 * @param[in] b Second 8 bit fixed point input vector
 *
 * @return The result of the 8 bit fixed point vector addition. The result is saturated in case of overflow
 */
qint8x16_t vqaddq_qs8(qint8x16_t a, qint8x16_t b);

/** 16 bit fixed point vector saturating add (8 elements)
 *
 * @param[in] a First 16 bit fixed point input vector
 * @param[in] b Second 16 bit fixed point input vector
 *
 * @return The result of the 16 bit fixed point vector addition. The result is saturated in case of overflow
 */
qint16x8_t vqaddq_qs16(qint16x8_t a, qint16x8_t b);

/** 8 bit fixed point vector saturating pairwise add (8 elements)
 *
 * @param[in] a 8 bit fixed point input vector
 *
 * @return The result of the 16 bit fixed point vector addition. The result is saturated in case of overflow
 */
int16x4_t vpaddl_qs8(qint8x8_t a);

/** 8 bit fixed point vector subtraction (8 elements)
 *
 * @param[in] a First 8 bit fixed point input vector
 * @param[in] b Second 8 bit fixed point input vector
 *
 * @return The result of the 8 bit fixed point vector subtraction
 */
qint8x8_t vsub_qs8(qint8x8_t a, qint8x8_t b);

/** 16 bit fixed point vector subtraction (4 elements)
 *
 * @param[in] a First 16 bit fixed point input vector
 * @param[in] b Second 16 bit fixed point input vector
 *
 * @return The result of the 16 bit fixed point vector subtraction
 */
qint16x4_t vsub_qs16(qint16x4_t a, qint16x4_t b);

/** 8 bit fixed point vector subtraction (16 elements)
 *
 * @param[in] a First 8 bit fixed point input vector
 * @param[in] b Second 8 bit fixed point input vector
 *
 * @return The result of the 8 bit fixed point vector subtraction
 */
qint8x16_t vsubq_qs8(qint8x16_t a, qint8x16_t b);

/** 16 bit fixed point vector subtraction (8 elements)
 *
 * @param[in] a First 16 bit fixed point input vector
 * @param[in] b Second 16 bit fixed point input vector
 *
 * @return The result of the 16 bit fixed point vector subtraction
 */
qint16x8_t vsubq_qs16(qint16x8_t a, qint16x8_t b);

/** 8 bit fixed point vector saturating subtraction (8 elements)
 *
 * @param[in] a First 8 bit fixed point input vector
 * @param[in] b Second 8 bit fixed point input vector
 *
 * @return The result of the 8 bit fixed point vector subtraction. The result is saturated in case of overflow
 */
qint8x8_t vqsub_qs8(qint8x8_t a, qint8x8_t b);

/** 16 bit fixed point vector saturating subtraction (4 elements)
 *
 * @param[in] a First 16 bit fixed point input vector
 * @param[in] b Second 16 bit fixed point input vector
 *
 * @return The result of the 16 bit fixed point vector subtraction. The result is saturated in case of overflow
 */
qint16x4_t vqsub_qs16(qint16x4_t a, qint16x4_t b);

/** 8 bit fixed point vector saturating subtraction (16 elements)
 *
 * @param[in] a First 8 bit fixed point input vector
 * @param[in] b Second 8 bit fixed point input vector
 *
 * @return The result of the 8 bit fixed point vector subtraction. The result is saturated in case of overflow
 */
qint8x16_t vqsubq_qs8(qint8x16_t a, qint8x16_t b);

/** 16 bit fixed point vector saturating subtraction (8 elements)
 *
 * @param[in] a First 16 bit fixed point input vector
 * @param[in] b Second 16 bit fixed point input vector
 *
 * @return The result of the 16 bit fixed point vector subtraction. The result is saturated in case of overflow
 */
qint16x8_t vqsubq_qs16(qint16x8_t a, qint16x8_t b);

/** 8 bit fixed point vector multiply (8 elements)
 *
 * @param[in] a                    First 8 bit fixed point input vector
 * @param[in] b                    Second 8 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8 bit fixed point vector multiplication.
 */
qint8x8_t vmul_qs8(qint8x8_t a, qint8x8_t b, int fixed_point_position);

/** 16 bit fixed point vector multiply (4 elements)
 *
 * @param[in] a                    First 16 bit fixed point input vector
 * @param[in] b                    Second 16 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit fixed point vector multiplication.
 */
qint16x4_t vmul_qs16(qint16x4_t a, qint16x4_t b, int fixed_point_position);

/** 8 bit fixed point vector multiply (16 elements)
 *
 * @param[in] a                    First 8 bit fixed point input vector
 * @param[in] b                    Second 8 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8 bit fixed point vector multiplication.
 */
qint8x16_t vmulq_qs8(qint8x16_t a, qint8x16_t b, int fixed_point_position);

/** 16 bit fixed point vector multiply (8 elements)
 *
 * @param[in] a                    First 16 bit fixed point input vector
 * @param[in] b                    Second 16 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit fixed point vector multiplication.
 */
qint16x8_t vmulq_qs16(qint16x8_t a, qint16x8_t b, int fixed_point_position);

/** 8 bit fixed point vector saturating multiply (8 elements)
 *
 * @param[in] a                    First 8 bit fixed point input vector
 * @param[in] b                    Second 8 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8 bit fixed point vector multiplication. The result is saturated in case of overflow
 */
qint8x8_t vqmul_qs8(qint8x8_t a, qint8x8_t b, int fixed_point_position);

/** 16 bit fixed point vector saturating multiply (4 elements)
 *
 * @param[in] a                    First 16 bit fixed point input vector
 * @param[in] b                    Second 16 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit fixed point vector multiplication. The result is saturated in case of overflow
 */
qint16x4_t vqmul_qs16(qint16x4_t a, qint16x4_t b, int fixed_point_position);

/** 8 bit fixed point vector saturating multiply (16 elements)
 *
 * @param[in] a                    First 8 bit fixed point input vector
 * @param[in] b                    Second 8 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8 bit fixed point vector multiplication. The result is saturated in case of overflow
 */
qint8x16_t vqmulq_qs8(qint8x16_t a, qint8x16_t b, int fixed_point_position);

/** 16 bit fixed point vector saturating multiply (8 elements)
 *
 * @param[in] a                    First 16 bit fixed point input vector
 * @param[in] b                    Second 16 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit fixed point vector multiplication. The result is saturated in case of overflow
 */
qint16x8_t vqmulq_qs16(qint16x8_t a, qint16x8_t b, int fixed_point_position);

/** 8 bit fixed point vector long multiply (8 elements)
 *
 * @param[in] a                    First 8 bit fixed point input vector
 * @param[in] b                    Second 8 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8 bit fixed point long vector multiplication.
 */
qint16x8_t vmull_qs8(qint8x8_t a, qint8x8_t b, int fixed_point_position);

/** 16 bit fixed point vector long multiply (4 elements)
 *
 * @param[in] a                    First 16 bit fixed point input vector
 * @param[in] b                    Second 16 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 32 bit fixed point long vector multiplication.
 */
qint32x4_t vmull_qs16(qint16x4_t a, qint16x4_t b, int fixed_point_position);

/** 8 bit fixed point vector multiply-accumulate (8 elements). This operation performs the product between @p b and @p c and add the result to @p a (a + b * c).
 *
 * @param[in] a                    First 8 bit fixed point input vector where the result of multiplication must be added to
 * @param[in] b                    Second 8 bit fixed point input vector
 * @param[in] c                    Third 8 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8 bit fixed point vector multiply-accumulate
 */
qint8x8_t vmla_qs8(qint8x8_t a, qint8x8_t b, qint8x8_t c, int fixed_point_position);

/** 16 bit fixed point vector multiply-accumulate (4 elements). This operation performs the product between @p b and @p c and add the result to @p a (a + b * c).
 *
 * @param[in] a                    First 16 bit fixed point input vector where the result of multiplication must be added to
 * @param[in] b                    Second 16 bit fixed point input vector
 * @param[in] c                    Third 16 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit fixed point vector multiply-accumulate
 */
qint16x4_t vmla_qs16(qint16x4_t a, qint16x4_t b, qint16x4_t c, int fixed_point_position);

/** 8 bit fixed point vector multiply-accumulate (16 elements). This operation performs the product between @p b and @p c and add the result to @p a (a + b * c).
 *
 * @param[in] a                    First 8 bit fixed point input vector where the result of multiplication must be added to
 * @param[in] b                    Second 8 bit fixed point input vector
 * @param[in] c                    Third 8 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8 bit fixed point vector multiply-accumulate
 */
qint8x16_t vmlaq_qs8(qint8x16_t a, qint8x16_t b, qint8x16_t c, int fixed_point_position);

/** 16 bit fixed point vector multiply-accumulate (16 elements). This operation performs the product between @p b and @p c and add the result to @p a (a + b * c).
 *
 * @param[in] a                    First 16 bit fixed point input vector where the result of multiplication must be added to
 * @param[in] b                    Second 16 bit fixed point input vector
 * @param[in] c                    Third 16 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit fixed point vector multiply-accumulate
 */
qint16x8_t vmlaq_qs16(qint16x8_t a, qint16x8_t b, qint16x8_t c, int fixed_point_position);

/** 8 bit fixed point vector saturating multiply-accumulate (8 elements). This operation performs the product between @p b and @p c and add the result to @p a (a + b * c).
 *
 * @param[in] a                    First 8 bit fixed point input vector where the result of multiplication must be added to
 * @param[in] b                    Second 8 bit fixed point input vector
 * @param[in] c                    Third 8 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8 bit fixed point vector multiply-accumulate. The result is saturated in case of overflow
 */
qint8x8_t vqmla_qs8(qint8x8_t a, qint8x8_t b, qint8x8_t c, int fixed_point_position);

/** 16 bit fixed point vector saturating multiply-accumulate (4 elements). This operation performs the product between @p b and @p c and add the result to @p a (a + b * c).
 *
 * @param[in] a                    First 16 bit fixed point input vector where the result of multiplication must be added to
 * @param[in] b                    Second 16 bit fixed point input vector
 * @param[in] c                    Third 16 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit fixed point vector multiply-accumulate. The result is saturated in case of overflow
 */
qint16x4_t vqmla_qs16(qint16x4_t a, qint16x4_t b, qint16x4_t c, int fixed_point_position);

/** 8 bit fixed point vector saturating multiply-accumulate (16 elements). This operation performs the product between @p b and @p c and add the result to @p a (a + b * c).
 *
 * @param[in] a                    First 8 bit fixed point input vector where the result of multiplication must be added to
 * @param[in] b                    Second 8 bit fixed point input vector
 * @param[in] c                    Third 8 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8 bit fixed point vector multiply-accumulate.The result is saturated in case of overflow
 */
qint8x16_t vqmlaq_qs8(qint8x16_t a, qint8x16_t b, qint8x16_t c, int fixed_point_position);

/** 16 bit fixed point vector saturating multiply-accumulate (8 elements). This operation performs the product between @p b and @p c and add the result to @p a (a + b * c).
 *
 * @param[in] a                    First 16 bit fixed point input vector where the result of multiplication must be added to
 * @param[in] b                    Second 16 bit fixed point input vector
 * @param[in] c                    Third 16 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit fixed point vector multiply-accumulate.The result is saturated in case of overflow
 */
qint16x8_t vqmlaq_qs16(qint16x8_t a, qint16x8_t b, qint16x8_t c, int fixed_point_position);

/** 8 bit fixed point vector multiply-accumulate long (8 elements).
 *  This operation performs the product between @p b and @p c and add the result to the 16 bit fixed point vector @p a (a + b * c). 8 elements
 *
 * @param[in] a                    First 16 bit fixed point input vector where the result of multiplication must be added to
 * @param[in] b                    Second 8 bit fixed point input vector
 * @param[in] c                    Third 8 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8 bit fixed point vector multiply-accumulate long
 */
qint16x8_t vmlal_qs8(qint16x8_t a, qint8x8_t b, qint8x8_t c, int fixed_point_position);

/** 16 bit fixed point vector multiply-accumulate long (4 elements).
 *  This operation performs the product between @p b and @p c and add the result to the 32 bit fixed point vector @p a (a + b * c). 4 elements
 *
 * @param[in] a                    First 32 bit fixed point input vector where the result of multiplication must be added to
 * @param[in] b                    Second 16 bit fixed point input vector
 * @param[in] c                    Third 16 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit fixed point vector multiply-accumulate long
 */
qint32x4_t vmlal_qs16(qint32x4_t a, qint16x4_t b, qint16x4_t c, int fixed_point_position);

/** 8 bit fixed point vector saturating multiply-accumulate long (8 elements). The saturation is performed on the 16 bit fixed point output vector.
 *  This operation performs the product between @p b and @p c and add the result to the 16 bit fixed point vector @p a (a + b * c). 8 elements
 *
 * @param[in] a                    First 16 bit fixed point input vector where the result of multiplication must be added to
 * @param[in] b                    Second 8 bit fixed point input vector
 * @param[in] c                    Third 8 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8 bit fixed point vector multiply-accumulate long
 */
qint16x8_t vqmlal_qs8(qint16x8_t a, qint8x8_t b, qint8x8_t c, int fixed_point_position);

/** 16 bit fixed point vector saturating multiply-accumulate long (4 elements). The saturation is performed on the 16 bit fixed point output vector.
 *  This operation performs the product between @p b and @p c and add the result to the 32 bit fixed point vector @p a (a + b * c). 4 elements
 *
 * @param[in] a                    First 32 bit fixed point input vector where the result of multiplication must be added to
 * @param[in] b                    Second 16 bit fixed point input vector
 * @param[in] c                    Third 16 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit fixed point vector multiply-accumulate long
 */
qint32x4_t vqmlal_qs16(qint32x4_t a, qint16x4_t b, qint16x4_t c, int fixed_point_position);

/** Convert a float vector with 4x2 elements to 8 bit fixed point vector with 8 elements
 *
 * @param[in] a                    Float input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the conversion float -> 8 bit fixed point. The result is saturated in case of overflow
 */
qint8x8_t vqcvt_qs8_f32(const float32x4x2_t a, int fixed_point_position);

/** Convert a float vector with 4 elements to 16 bit fixed point vector with 4 elements
 *
 * @param[in] a                    Float input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the conversion float -> 16 bit fixed point. The result is saturated in case of overflow
 */
qint16x4_t vqcvt_qs16_f32(const float32x4_t a, int fixed_point_position);

/** Convert a float vector with 4x4 elements to 8 bit fixed point vector with 16 elements
 *
 * @param[in] a                    Float input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the conversion float -> 8 bit fixed point. The result is saturated in case of overflow
 */
qint8x16_t vqcvtq_qs8_f32(const float32x4x4_t &a, int fixed_point_position);

/** Convert a float vector with 4x2 elements to 16 bit fixed point vector with 8 elements
 *
 * @param[in] a                    Float input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the conversion float -> 16 bit fixed point. The result is saturated in case of overflow
 */
qint16x8_t vqcvtq_qs16_f32(const float32x4x2_t &a, int fixed_point_position);

/** Convert a 8 bit fixed point vector with 8 elements to a float vector with 4x2 elements
 *
 * @param[in] a                    8 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the conversion 8 bit fixed point -> float32x2x4
 */
float32x4x2_t vcvt_f32_qs8(qint8x8_t a, int fixed_point_position);

/** Convert a 16 bit fixed point vector with 4 elements to a float vector with 4 elements
 *
 * @param[in] a                    16 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the conversion 16 bit fixed point -> float32x2
 */
float32x4_t vcvt_f32_qs16(qint16x4_t a, int fixed_point_position);

/** Convert a 8 bit fixed point vector with 16 elements to a float vector with 4x4 elements
 *
 * @param[in] a                    8 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the conversion 8 bit fixed point -> float32x4x4
 */
float32x4x4_t vcvtq_qs8_f32(qint8x16_t a, int fixed_point_position);

/** Convert a 16 bit fixed point vector with 8 elements to a float vector with 4x2 elements
 *
 * @param[in] a                    16 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the conversion 16 bit fixed point -> float32x4x2
 */
float32x4x2_t vcvtq_qs16_f32(qint16x8_t a, int fixed_point_position);

/** Calculate reciprocal of a fixed point 8bit number using the Newton-Raphson method. (8 elements)
 *
 * @param[in] a                    8bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8bit reciprocal (1/a).
 */
qint8x8_t vrecip_qs8(qint8x8_t a, int fixed_point_position);

/** Calculate reciprocal of a fixed point 8bit number using the Newton-Raphson method. (4 elements)
 *
 * @param[in] a                    16 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit reciprocal (1/a).
 */
qint16x4_t vrecip_qs16(qint16x4_t a, int fixed_point_position);

/** Calculate reciprocal of a fixed point 8bit number using the Newton-Raphson method. (16 elements)
 *
 * @param[in] a                    8bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8bit reciprocal (1/a).
 */
qint8x16_t vrecipq_qs8(qint8x16_t a, int fixed_point_position);

/** Calculate reciprocal of a fixed point 8bit number using the Newton-Raphson method. (8 elements)
 *
 * @param[in] a                    16 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit reciprocal (1/a).
 */
qint16x8_t vrecipq_qs16(qint16x8_t a, int fixed_point_position);

/** Division fixed point 8bit (8 elements)
 *
 * @param[in] a                    First 8bit fixed point input vector
 * @param[in] b                    Second 8bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The quotient and remainder number in fixed point format.
 */
qint8x8_t vdiv_qs8(qint8x8_t a, int8x8_t b, int fixed_point_position);

/** Division fixed point 16 bit (4 elements)
 *
 * @param[in] a                    First 16 bit fixed point input vector
 * @param[in] b                    Second  16 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The quotient and remainder number in fixed point format.
 */
qint16x4_t vdiv_qs16(qint16x4_t a, qint16x4_t b, int fixed_point_position);

/** Division fixed point 8bit (16 elements)
 *
 * @param[in] a                    First 8bit fixed point input vector
 * @param[in] b                    Second 8bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The quotient and remainder number in 8bit fixed point format.
 */
qint8x16_t vdivq_qs8(qint8x16_t a, qint8x16_t b, int fixed_point_position);

/** Division fixed point 16 bit (8 elements)
 *
 * @param[in] a                    First 16 bit fixed point input vector
 * @param[in] b                    Second 16 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The quotient and remainder number in 16 bit fixed point format.
 */
qint16x8_t vdivq_qs16(qint16x8_t a, qint16x8_t b, int fixed_point_position);

/** Perform a 4th degree polynomial approximation. (8 elements)
 *
 * @param[in] a                    8bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8bit taylor approximation.
 */
template <bool islog>
qint8x8_t vtaylor_poly_qs8(qint8x8_t a, int fixed_point_position);

/** Perform a 4th degree polynomial approximation. (4 elements)
 *
 * @param[in] a                    16 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit taylor approximation.
 */
template <bool islog>
qint16x4_t vtaylor_poly_qs16(qint16x4_t a, int fixed_point_position);

/** Perform a 4th degree polynomial approximation. (16 elements)
 *
 * @param[in] a                    8bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8bit taylor approximation.
 */
template <bool islog>
qint8x16_t vtaylor_polyq_qs8(qint8x16_t a, int fixed_point_position);

/** Perform a 4th degree polynomial approximation. (8 elements)
 *
 * @param[in] a                    16 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8bit taylor approximation.
 */
template <bool islog>
qint16x8_t vtaylor_polyq_qs16(qint16x8_t a, int fixed_point_position);

/** Calculate saturating exponential fixed point 8bit (8 elements)
 *
 * @param[in] a                    8bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8bit saturating exponential
 */
qint8x8_t vqexp_qs8(qint8x8_t a, int fixed_point_position);

/** Calculate saturating exponential fixed point 16 bit (4 elements)
 *
 * @param[in] a                    8bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit saturating exponential
 */
qint16x4_t vqexp_qs16(qint16x4_t a, int fixed_point_position);

/** Calculate saturating exponential fixed point 8bit (16 elements)
 *
 * @param[in] a                    8bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8bit saturating exponential
 */
qint8x16_t vqexpq_qs8(qint8x16_t a, int fixed_point_position);

/** Calculate saturating exponential fixed point 16 bit (8 elements)
 *
 * @param[in] a                    16 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit saturating exponential
 */
qint16x8_t vqexpq_qs16(qint16x8_t a, int fixed_point_position);

/** Calculate logarithm fixed point 8 bit (8 elements)
 *
 * @param[in] a                    8bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8bit logarithm.
 */
qint8x8_t vlog_qs8(qint8x8_t a, int fixed_point_position);

/** Calculate logarithm fixed point 16 bit (4 elements)
 *
 * @param[in] a                    16 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit logarithm.
 */
qint16x4_t vlog_qs16(qint16x4_t a, int fixed_point_position);

/** Calculate logarithm fixed point 16bit (16 elements)
 *
 * @param[in] a                    8bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8bit logarithm.
 */
qint8x16_t vlogq_qs8(qint8x16_t a, int fixed_point_position);

/** Calculate logarithm fixed point 16 bit (8 elements)
 *
 * @param[in] a                    16 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit logarithm.
 */
qint16x8_t vlogq_qs16(qint16x8_t a, int fixed_point_position);

/** Calculate inverse square root for fixed point 8bit using Newton-Raphosn method (8 elements)
 *
 * @param[in] a                    8bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8bit inverse sqrt.
 */
qint8x8_t vinvsqrt_qs8(qint8x8_t a, int fixed_point_position);

/** Calculate inverse square root for fixed point 16 bit using Newton-Raphosn method (4 elements)
 *
 * @param[in] a                    16 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit inverse sqrt.
 */
qint16x4_t vinvsqrt_qs16(qint16x4_t a, int fixed_point_position);

/** Calculate saturating inverse square root for fixed point 8bit using Newton-Raphosn method (8 elements)
 *
 * @param[in] a                    8bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8bit inverse sqrt.
 */
qint8x8_t vqinvsqrt_qs8(qint8x8_t a, int fixed_point_position);

/** Calculate saturating inverse square root for fixed point 16 bit using Newton-Raphosn method (4 elements)
 *
 * @param[in] a                    16 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit inverse sqrt.
 */
qint16x4_t vqinvsqrt_qs16(qint16x4_t a, int fixed_point_position);

/** Calculate inverse square root for fixed point 8bit using Newton-Raphosn method (16 elements)
 *
 * @param[in] a                    8bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8bit inverse sqrt.
 */
qint8x16_t vinvsqrtq_qs8(qint8x16_t a, int fixed_point_position);

/** Calculate inverse square root for fixed point 8bit using Newton-Raphosn method (8 elements)
 *
 * @param[in] a                    16 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit inverse sqrt.
 */
qint16x8_t vinvsqrtq_qs16(qint16x8_t a, int fixed_point_position);

/** Calculate saturating inverse square root for fixed point 8bit using Newton-Raphosn method (16 elements)
 *
 * @param[in] a                    8bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8bit inverse sqrt.
 */
qint8x16_t vqinvsqrtq_qs8(qint8x16_t a, int fixed_point_position);

/** Calculate saturating inverse square root for fixed point 16 bit using Newton-Raphosn method (8 elements)
 *
 * @param[in] a                    16 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit inverse sqrt.
 */
qint16x8_t vqinvsqrtq_qs16(qint16x8_t a, int fixed_point_position);

/** Calculate hyperbolic tangent for fixed point 8bit (8 elements)
 *
 * @param[in] a                    8bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The calculated Hyperbolic Tangent.
 */
qint8x8_t vqtanh_qs8(qint8x8_t a, int fixed_point_position);

/** Calculate hyperbolic tangent for fixed point 16 bit (4 elements)
 *
 * @param[in] a                    16 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The calculated Hyperbolic Tangent.
 */
qint16x4_t vqtanh_qs16(qint16x4_t a, int fixed_point_position);

/** Calculate hyperbolic tangent for fixed point 8bit (16 elements)
 *
 * @param[in] a                    8bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The calculated Hyperbolic Tangent.
 */
qint8x16_t vqtanhq_qs8(qint8x16_t a, int fixed_point_position);

/** Calculate hyperbolic tangent for fixed point 16bit (8 elements)
 *
 * @param[in] a                    16 bit fixed point input vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The calculated Hyperbolic Tangent.
 */
qint16x8_t vqtanhq_qs16(qint16x8_t a, int fixed_point_position);

/** Calculate saturating n power for fixed point 8bit (16 elements).
 *
 * pow(a,b) = e^(b*log(a))
 *
 * @param[in] a                    8bit fixed point input vector
 * @param[in] b                    8bit fixed point power vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8bit power.
 */
qint8x16_t vqpowq_qs8(qint8x16_t a, qint8x16_t b, int fixed_point_position);

/** Calculate saturating n power for fixed point 16bit (8 elements).
 *
 * pow(a,b) = e^(b*log(a))
 *
 * @param[in] a                    16bit fixed point input vector
 * @param[in] b                    16bit fixed point power vector
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16bit power.
 */
qint16x8_t vqpowq_qs16(qint16x8_t a, qint16x8_t b, int fixed_point_position);

/** Compute lane-by-lane maximum between elements of a float vector with 4x2 elements
 *
 * @param[in] a Float input vector
 * @param[in] b Float input vector
 *
 * @return The lane-by-lane maximum -> float32x4x2
 */
float32x4x2_t vmax2q_f32(float32x4x2_t a, float32x4x2_t b);
} // namespace arm_compute
#include "arm_compute/core/NEON/NEFixedPoint.inl"
#endif /* __ARM_COMPUTE_NEFIXEDPOINT_H__ */
