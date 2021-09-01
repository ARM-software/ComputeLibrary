/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_SIZE2D_H
#define ARM_COMPUTE_SIZE2D_H

#include <cstddef>
#include <string>
#include <utility>

namespace arm_compute
{
/** Class for specifying the size of an image or rectangle */
class Size2D
{
public:
    /** Default constructor */
    Size2D() = default;
    /** Constructor. Initializes "width" and "height" respectively with "w" and "h"
     *
     * @param[in] w Width of the image or rectangle
     * @param[in] h Height of the image or rectangle
     */
    Size2D(size_t w, size_t h) noexcept
        : width(w),
          height(h)
    {
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

    bool operator==(const Size2D &other) const
    {
        return (width == other.width) && (height == other.height);
    }

    bool operator!=(const Size2D &other) const
    {
        return !(*this == other);
    }

    std::string to_string() const;

    /** Semantic accessor for width as x.
     *
     * @return x.
     */
    size_t x() const
    {
        return width;
    }

    /** Semantic accessor for height as y.
     *
     * @return y.
     */
    size_t y() const
    {
        return height;
    }

public:
    size_t width  = {}; /**< Width of the image region or rectangle */
    size_t height = {}; /**< Height of the image region or rectangle */
};
}
#endif /*ARM_COMPUTE_SIZE2D_H */
