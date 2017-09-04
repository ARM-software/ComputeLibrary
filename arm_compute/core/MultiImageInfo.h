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
#ifndef __ARM_COMPUTE_MULTIIMAGEINFO_H__
#define __ARM_COMPUTE_MULTIIMAGEINFO_H__

#include "arm_compute/core/Types.h"

namespace arm_compute
{
/** Store the multi-planar image's metadata */
class MultiImageInfo
{
public:
    /** Constructor */
    MultiImageInfo();
    /** Initialize the metadata structure with the given parameters
     *
     * @param[in] width  Width of the image (in number of pixels)
     * @param[in] height Height of the image (in number of pixels)
     * @param[in] format Colour format of the image.
     */
    void init(unsigned int width, unsigned int height, Format format);
    /** Colour format of the image
     *
     * @return Colour format of the image
     */
    Format format() const;
    /** Width in pixels
     *
     * @return The width in pixels
     */
    unsigned int width() const;
    /** Height in pixels
     *
     * @return The height in pixels
     */
    unsigned int height() const;

protected:
    unsigned int _width;
    unsigned int _height;
    Format       _format;
};
}
#endif /*__ARM_COMPUTE_MULTIIMAGEINFO_H__ */
