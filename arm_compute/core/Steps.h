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
#ifndef ARM_COMPUTE_STEPS_H
#define ARM_COMPUTE_STEPS_H

#include "arm_compute/core/Dimensions.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Types.h"

#include <algorithm>
#include <array>
#include <cstddef>

namespace arm_compute
{
/** Class to describe a number of elements in each dimension. Similar to @ref
 *  Strides but not in bytes but number of elements.
 */
class Steps : public Dimensions<uint32_t>
{
public:
    /** Constructor to initialize the steps.
     *
     * @param[in] steps Values to initialize the steps.
     */
    template <typename... Ts>
    Steps(Ts... steps)
        : Dimensions{ steps... }
    {
        // Initialize empty dimensions to 1
        std::fill(_id.begin() + _num_dimensions, _id.end(), 1);
    }
    /** Allow instances of this class to be copy constructed */
    constexpr Steps(const Steps &) = default;
    /** Allow instances of this class to be copied */
    Steps &operator=(const Steps &) = default;
    /** Allow instances of this class to be move constructed */
    constexpr Steps(Steps &&) = default;
    /** Allow instances of this class to be moved */
    Steps &operator=(Steps &&) = default;
    /** Default destructor */
    ~Steps() = default;
};
}
#endif /*ARM_COMPUTE_STEPS_H*/
