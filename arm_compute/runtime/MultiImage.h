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
#ifndef __ARM_COMPUTE_MULTIIMAGE_H__
#define __ARM_COMPUTE_MULTIIMAGE_H__

#include "arm_compute/core/IMultiImage.h"
#include "arm_compute/core/MultiImageInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"

#include <array>

namespace arm_compute
{
class Coordinates;
class ITensor;
using IImage = ITensor;

/** Basic implementation of the multi-planar image interface */
class MultiImage : public IMultiImage
{
public:
    /** Constructor */
    MultiImage();
    /** Allocate the multi-planar image
     *
     * @param[in] width  Width of the whole image
     * @param[in] height Height of the whole image
     * @param[in] format Format of the whole image
     */
    void init(unsigned int width, unsigned int height, Format format);
    /** Allocate the multi-planar image
     *
     * @note Uses conservative padding strategy which fits all kernels.
     *
     * @param[in] width  Width of the whole image
     * @param[in] height Height of the whole image
     * @param[in] format Format of the whole image
     */
    void init_auto_padding(unsigned int width, unsigned int height, Format format);
    /** Allocated a previously initialised multi image
     *
     * @note The multi image must not already be allocated when calling this function.
     *
     **/
    void allocate();
    /** Create a subimage from an existing MultiImage.
     *
     * @param[in] image  Image to use backing memory from
     * @param[in] coords Starting coordinates of the new image. Should be within the parent image sizes
     * @param[in] width  The width of the subimage
     * @param[in] height The height of the subimage
     */
    void create_subimage(MultiImage *image, const Coordinates &coords, unsigned int width, unsigned int height);

    // Inherited methods overridden:
    const MultiImageInfo *info() const override;
    Image *plane(unsigned int index) override;
    const Image *plane(unsigned int index) const override;

private:
    /** Init the multi-planar image
     *
     * @param[in] width        Width of the whole image
     * @param[in] height       Height of the whole image
     * @param[in] format       Format of the whole image
     * @param[in] auto_padding Specifies whether the image uses auto padding
     */
    void internal_init(unsigned int width, unsigned int height, Format format, bool auto_padding);

    MultiImageInfo _info;        /** Instance of the multi-planar image's meta data */
    std::array<Image, 3> _plane; /* Instance Image to hold the planar's information */
};
}
#endif /*__ARM_COMPUTE_MULTIIMAGE_H__ */
