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
#ifndef __ARM_COMPUTE_IMULTIIMAGE_H__
#define __ARM_COMPUTE_IMULTIIMAGE_H__

namespace arm_compute
{
class ITensor;
using IImage = ITensor;
class MultiImageInfo;

/** Interface for multi-planar images */
class IMultiImage
{
public:
    /** Destructor */
    virtual ~IMultiImage() = default;
    /** Interface to be implemented by the child class to return the multi-planar image's metadata
     *
     * @return A pointer to the image's metadata.
     */
    virtual const MultiImageInfo *info() const = 0;
    /** Return a pointer to the requested plane of the image.
     *
     *  @param[in] index The index of the wanted planed.
     *
     *  @return A pointer pointed to the plane
     */
    virtual IImage *plane(unsigned int index) = 0;
    /** Return a constant pointer to the requested plane of the image.
     *
     *  @param[in] index The index of the wanted planed.
     *
     *  @return A constant pointer pointed to the plane
     */
    virtual const IImage *plane(unsigned int index) const = 0;
};
}
#endif /*__ARM_COMPUTE_IMULTIIMAGE_H__ */
