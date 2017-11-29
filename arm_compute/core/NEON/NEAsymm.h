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
#ifndef __ARM_COMPUTE_NEASYMM_H__
#define __ARM_COMPUTE_NEASYMM_H__

#include <arm_neon.h>

namespace arm_compute
{
using qasymm8x8_t   = uint8x8_t;   /**< 8 bit quantized asymmetric vector with 8 elements */
using qasymm8x8x2_t = uint8x8x2_t; /**< 8 bit quantized asymmetric vector with 16 elements */
using qasymm8x8x3_t = uint8x8x3_t; /**< 8 bit quantized asymmetric vector with 24 elements */
using qasymm8x8x4_t = uint8x8x4_t; /**< 8 bit quantized asymmetric vector with 32 elements */
using qasymm8x16_t  = uint8x16_t;  /**< 8 bit quantized asymmetric vector with 16 elements */

/** Round to the nearest division by a power-of-two using exponent
 *
 * @note This function calculates the following expression: (x + 2^n -1 ) / 2^n where n = exponent
 *
 * @param[in] x        Vector of 4 elements
 * @param[in] exponent Integer value used to round to nearest division by a power-of-two
 *
 * @return the nearest division by a power-of-two using exponent
 */
int32x4_t rounding_divide_by_pow2(int32x4_t x, int exponent);

/** Perform a multiply-accumulate on all 16 components of a QASYMM8 vector
 *
 * vd*vs + vo
 *
 * @param[in] vd Input vector value in QASYMM8 format
 * @param[in] vs Vector multiplier in F32 format. The multiplier value must be duplicated across all four lanes.
 * @param[in] vo Vector addend in F32 format. The addend value must be duplicated across all four lanes.
 *
 * @return A 16-component vector in QASYMM8 format, saturated to fit
 */
uint8x16_t vmlaq_qasymm8(qasymm8x16_t vd, float32x4_t vs, float32x4_t vo);
} // namespace arm_compute
#include "arm_compute/core/NEON/NEAsymm.inl"
#endif // __ARM_COMPUTE_NEASYMM_H__
