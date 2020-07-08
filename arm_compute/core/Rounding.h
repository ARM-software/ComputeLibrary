/*
 * Copyright (c) 2017-2019 Arm Limited.
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
#ifndef ARM_COMPUTE_ROUNDING_H
#define ARM_COMPUTE_ROUNDING_H

namespace arm_compute
{
/** Rounding method */
enum class RoundingPolicy
{
    TO_ZERO,         /**< Truncates the least significant values that are lost in operations. */
    TO_NEAREST_UP,   /**< Rounds to nearest value; half rounds away from zero */
    TO_NEAREST_EVEN, /**< Rounds to nearest value; half rounds to nearest even */
};

/** Return a rounded value of x. Rounding is done according to the rounding_policy.
 *
 * @param[in] x               Float value to be rounded.
 * @param[in] rounding_policy Policy determining how rounding is done.
 *
 * @return Rounded value of the argument x.
 */
int round(float x, RoundingPolicy rounding_policy);
}
#endif /*ARM_COMPUTE_ROUNDING_H */
