/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef ARM_COMPUTE_SIZE3D_H
#define ARM_COMPUTE_SIZE3D_H

#include <string>

namespace arm_compute
{
/** Class for specifying the size of a 3D shape or object */
class Size3D
{
public:
    /** Default constructor */
    Size3D() = default;
    /** Constructor. Initializes "width", "height" and "depth" respectively with "w", "h" and "d"
     *
     * @param[in] w Width of the 3D shape or object
     * @param[in] h Height of the 3D shape or object
     * @param[in] d Depth of the 3D shape or object
     */
    Size3D(size_t w, size_t h, size_t d) noexcept
        : width(w), height(h), depth(d)
    {
    }

    /** Convert the values stored to string
     *
     * @return string of (width x height x depth).
     */
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

    /** Semantic accessor for depth as z.
     *
     * @return z.
     */
    size_t z() const
    {
        return depth;
    }

    bool operator!=(const Size3D &other) const
    {
        return !(*this == other);
    }

    bool operator==(const Size3D &other) const
    {
        return (width == other.width) && (height == other.height) && (depth == other.depth);
    }

public:
    size_t width  = {}; /**< Width of the 3D shape or object */
    size_t height = {}; /**< Height of the 3D shape or object */
    size_t depth  = {}; /**< Depth of the 3D shape or object */
};

} // namespace arm_compute
#endif /* ARM_COMPUTE_SIZE3D_H */
