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
#ifndef __ARM_COMPUTE_STRIDES_H__
#define __ARM_COMPUTE_STRIDES_H__

#include "arm_compute/core/Dimensions.h"
#include "arm_compute/core/Error.h"

#include <algorithm>
#include <array>
#include <cstddef>

namespace arm_compute
{
/** Strides of an item in bytes */
class Strides : public Dimensions<size_t>
{
public:
    /** Constructor to initialize the strides.
     *
     * @param[in] strides Values to initialize the strides.
     */
    template <typename... Ts>
    constexpr Strides(Ts... strides)
        : Dimensions{ strides... }
    {
    }
    /** Allow instances of this class to be copy constructed */
    constexpr Strides(const Strides &) = default;
    /** Allow instances of this class to be copied */
    Strides &operator=(const Strides &) = default;
    /** Allow instances of this class to be move constructed */
    constexpr Strides(Strides &&) = default;
    /** Allow instances of this class to be moved */
    Strides &operator=(Strides &&) = default;
    /** Default destructor */
    ~Strides() = default;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_STRIDES_H__*/
