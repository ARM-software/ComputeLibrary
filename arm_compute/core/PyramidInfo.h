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
#ifndef __ARM_COMPUTE_PYRAMIDINFO_H__
#define __ARM_COMPUTE_PYRAMIDINFO_H__

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

#include <cstddef>

namespace arm_compute
{
/** Store the Pyramid's metadata */
class PyramidInfo
{
public:
    /** Default constructor */
    PyramidInfo();
    /** Default destructor */
    virtual ~PyramidInfo() = default;
    /** Allow instances of this class to be copy constructed */
    PyramidInfo(const PyramidInfo &) = default;
    /** Allow instances of this class to be copied */
    PyramidInfo &operator=(const PyramidInfo &) = default;
    /** Allow instances of this class to be move constructed */
    PyramidInfo(PyramidInfo &&) = default;
    /** Allow instances of this class to be moved */
    PyramidInfo &operator=(PyramidInfo &&) = default;

    /** Create pyramid info for 2D tensors
     *
     * @param[in] num_levels The number of pyramid levels. This is required to be a non-zero value
     * @param[in] scale      Used to indicate the scale between the pyramid levels.
     *                       This is required to be a non-zero positive value.
     * @param[in] width      The width of the 2D tensor at 0th pyramid level
     * @param[in] height     The height of the 2D tensor at 0th pyramid level
     * @param[in] format     The format of all 2D tensors in the pyramid
     *                       NV12, NV21, IYUV, UYVY and YUYV formats are not supported.
     */
    PyramidInfo(size_t num_levels, float scale, size_t width, size_t height, Format format);

    /** Create pyramid info using TensorShape
     *
     * @param[in] num_levels   The number of pyramid levels. This is required to be a non-zero value
     * @param[in] scale        Used to indicate the scale between the pyramid levels.
     *                         This is required to be a non-zero positive value.
     * @param[in] tensor_shape It specifies the size for each dimension of the tensor 0th pyramid level in number of elements
     * @param[in] format       The format of all tensors in the pyramid
     */
    PyramidInfo(size_t num_levels, float scale, const TensorShape &tensor_shape, Format format);

    /** Initialize pyramid's metadata for 2D tensors
     *
     * @param[in] num_levels The number of pyramid levels. This is required to be a non-zero value
     * @param[in] scale      Used to indicate the scale between the pyramid levels.
     *                       This is required to be a non-zero positive value.
     * @param[in] width      The width of the 2D tensor at 0th pyramid level
     * @param[in] height     The height of the 2D tensor at 0th pyramid level
     * @param[in] format     The format of all 2D tensors in the pyramid
     *                       NV12, NV21, IYUV, UYVY and YUYV formats are not supported.
     */
    void init(size_t num_levels, float scale, size_t width, size_t height, Format format);
    /** Initialize pyramid's metadata using TensorShape
     *
     * @param[in] num_levels   The number of pyramid levels. This is required to be a non-zero value
     * @param[in] scale        Used to indicate the scale between the pyramid levels.
     *                         This is required to be a non-zero positive value.
     * @param[in] tensor_shape It specifies the size for each dimension of the tensor 0th pyramid level in number of elements
     * @param[in] format       The format of all tensors in the pyramid
     */
    void init(size_t num_levels, float scale, const TensorShape &tensor_shape, Format format);
    /** Return the number of the pyramid levels
     *
     *  @return The number of the pyramid levels
     */
    size_t num_levels() const;
    /** Return the width of the 0th level tensor
     *
     *  @return The width of the 0th level tensor
     */
    size_t width() const;
    /** Return the height of the 0th level tensor
     *
     *  @return The height of the 0th level tensor
     */
    size_t height() const;
    /** Return the TensorShape of the o-th level tensor
     *
     * @return
     */
    const TensorShape &tensor_shape() const;
    /** Return the image format of all tensor in the pyramid
     *
     *  @return The image format
     */
    Format format() const;
    /** Return the scale factor of the pyramid
     *
     *  @return Return the scale factor
     */
    float scale() const;

private:
    size_t      _num_levels;
    TensorShape _tensor_shape;
    Format      _format;
    float       _scale;
};
}
#endif /*__ARM_COMPUTE_PYRAMIDINFO_H__ */
