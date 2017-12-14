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
#ifndef __ARM_COMPUTE_ICLMULTIIMAGE_H__
#define __ARM_COMPUTE_ICLMULTIIMAGE_H__

#include "arm_compute/core/IMultiImage.h"

namespace arm_compute
{
class ICLTensor;
using ICLImage = ICLTensor;

/** Interface for OpenCL multi-planar images */
class ICLMultiImage : public IMultiImage
{
public:
    /** Return a pointer to the requested OpenCL plane of the image.
     *
     * @param[in] index The index of the wanted planed.
     *
     *  @return A pointer pointed to the OpenCL plane
     */
    virtual ICLImage *cl_plane(unsigned int index) = 0;
    /** Return a constant pointer to the requested OpenCL plane of the image.
     *
     * @param[in] index The index of the wanted planed.
     *
     *  @return A constant pointer pointed to the OpenCL plane
     */
    virtual const ICLImage *cl_plane(unsigned int index) const = 0;

    // Inherited methods overridden:
    IImage *plane(unsigned int index) override;
    const IImage *plane(unsigned int index) const override;
};
}
#endif /*__ARM_COMPUTE_ICLMULTIIMAGE_H__ */
