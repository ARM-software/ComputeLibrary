/*
 * Copyright (c) 2016-2023 Arm Limited.
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
#ifndef ARM_COMPUTE_UTILS_ADJUSTVECSIZE_H
#define ARM_COMPUTE_UTILS_ADJUSTVECSIZE_H

#include "arm_compute/core/Error.h"

namespace arm_compute
{
/** Returns the adjusted vector size in case it is less than the input's first dimension, getting rounded down to its closest valid vector size
 *
 * @param[in] vec_size vector size to be adjusted
 * @param[in] dim0     size of the first dimension
 *
 * @return the number of element processed along the X axis per thread
 */
inline unsigned int adjust_vec_size(unsigned int vec_size, size_t dim0)
{
    ARM_COMPUTE_ERROR_ON(vec_size > 16);

    if((vec_size >= dim0) && (dim0 == 3))
    {
        return dim0;
    }

    while(vec_size > dim0)
    {
        vec_size >>= 1;
    }

    return vec_size;
}
}
#endif /*ARM_COMPUTE_UTILS_H */
