/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_SIZE2D_H__
#define __ARM_COMPUTE_SIZE2D_H__

#include <cstddef>

namespace arm_compute
{
/** Class for specifying the size of an image or rectangle */
class Size2D
{
public:
    /** Default constructor */
    Size2D()
        : width(0), height(0)
    {
    }
    /** Constructor. Initializes "width" and "height" respectively with "w" and "h"
     *
     * @param[in] w Width of the image or rectangle
     * @param[in] h Height of the image or rectangle
     */
    Size2D(size_t w, size_t h)
        : width(w), height(h)
    {
    }
    /** Constructor. Initializes "width" and "height" with the dimensions of "size"
     *
     * @param[in] size Size data object
     */
    Size2D(const Size2D &size)
        : width(size.width), height(size.height)
    {
    }
    /** Copy assignment
     *
     * @param[in] size Constant reference input "Size2D" data object to copy
     *
     * @return Reference to the newly altered left hand side "Size2D" data object
     */
    Size2D &operator=(const Size2D &size)
    {
        width  = size.width;
        height = size.height;
        return *this;
    }
    /** The area of the image or rectangle calculated as (width * height)
     *
     * @return Area (width * height)
     *
     */
    size_t area() const
    {
        return (width * height);
    }

public:
    size_t width;  /**< Width of the image region or rectangle */
    size_t height; /**< Height of the image region or rectangle */
};
}
#endif /*__ARM_COMPUTE_SIZE2D_H__ */
