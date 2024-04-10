/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_NEFIXEDPOINT_H
#define ARM_COMPUTE_NEFIXEDPOINT_H

#include <arm_neon.h>

namespace arm_compute
{
/** Compute lane-by-lane maximum between elements of a float vector with 4x2 elements
 *
 * @param[in] a Float input vector
 * @param[in] b Float input vector
 *
 * @return The lane-by-lane maximum -> float32x4x2
 */
float32x4x2_t vmax2q_f32(float32x4x2_t a, float32x4x2_t b);
} // namespace arm_compute
#include "src/core/NEON/NEFixedPoint.inl"
#endif /* ARM_COMPUTE_NEFIXEDPOINT_H */