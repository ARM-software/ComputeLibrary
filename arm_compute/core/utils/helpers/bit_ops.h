/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_UTILS_HELPERS_BIT_OPS_H__
#define __ARM_COMPUTE_UTILS_HELPERS_BIT_OPS_H__

#include "arm_compute/core/utils/misc/Requires.h"

#include <type_traits>

namespace arm_compute
{
namespace helpers
{
namespace bit_ops
{
/** Checks if the idx-th bit is set in an integral type
 *
 * @param[in] v   Integral input
 * @param[in] idx Index of the bit to check
 *
 * @return True if the idx-th bit is set else false
 */
template <typename T, REQUIRES_TA(std::is_integral<T>::value)>
bool is_bit_set(T v, unsigned int idx)
{
    return (v & 1 << idx) != 0;
}
} // namespace bit_ops
} // namespace helpers
} // namespace arm_compute
#endif /* __ARM_COMPUTE_UTILS_HELPERS_BIT_OPS_H__ */
