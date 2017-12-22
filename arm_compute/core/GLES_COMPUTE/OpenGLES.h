/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_OPENGLES_H__
#define __ARM_COMPUTE_OPENGLES_H__

#include "arm_compute/core/Log.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <EGL/eglplatform.h>
#include <GLES3/gl31.h>
#include <GLES3/gl3ext.h>
#include <cstddef>

#ifdef ARM_COMPUTE_DEBUG_ENABLED
#define ARM_COMPUTE_GL_CHECK(x)                                                                      \
    x;                                                                                               \
    {                                                                                                \
        GLenum error = glGetError();                                                                 \
        if(error != GL_NO_ERROR)                                                                     \
            ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("glGetError() = %i (0x%.8x)\n", error, error); \
    }
#else /* ARM_COMPUTE_DEBUG_ENABLED */
#define ARM_COMPUTE_GL_CHECK(x) x
#endif /* ARM_COMPUTE_DEBUG_ENABLED */

namespace arm_compute
{
namespace gles
{
/** Class interface for specifying NDRange values. */
class NDRange
{
private:
    size_t _sizes[3];
    size_t _dimensions;

public:
    /** Default constructor - resulting range has zero dimensions. */
    NDRange()
        : _dimensions(0)
    {
        _sizes[0] = 0;
        _sizes[1] = 0;
        _sizes[2] = 0;
    }

    /** Constructs one-dimensional range.
     *
     * @param[in] size0 Size of the first dimension.
     */
    NDRange(size_t size0)
        : _dimensions(1)
    {
        _sizes[0] = size0;
        _sizes[1] = 1;
        _sizes[2] = 1;
    }

    /** Constructs two-dimensional range.
     *
     * @param[in] size0 Size of the first dimension.
     * @param[in] size1 Size of the second dimension.
     */
    NDRange(size_t size0, size_t size1)
        : _dimensions(2)
    {
        _sizes[0] = size0;
        _sizes[1] = size1;
        _sizes[2] = 1;
    }

    /** Constructs three-dimensional range.
     *
     * @param[in] size0 Size of the first dimension.
     * @param[in] size1 Size of the second dimension.
     * @param[in] size2 Size of the third dimension.
     */
    NDRange(size_t size0, size_t size1, size_t size2)
        : _dimensions(3)
    {
        _sizes[0] = size0;
        _sizes[1] = size1;
        _sizes[2] = size2;
    }

    /** Conversion operator to const size_t *.
     *
     *  @returns A pointer to the size of the first dimension.
     */
    operator const size_t *() const
    {
        return _sizes;
    }

    /** Queries the number of dimensions in the range.
     *
     * @returns The number of dimensions.
    */
    size_t dimensions() const
    {
        return _dimensions;
    }

    /** Returns the size of the object in bytes based on the runtime number of dimensions
     *
     * @returns The size of the object in bytes.
     */
    size_t size() const
    {
        return _dimensions * sizeof(size_t);
    }

    /** Returns the sizes array for each dimensions.
     *
     * @returns The sizes array
     */
    size_t *get()
    {
        return _sizes;
    }

    /** Returns the sizes array for each dimensions.
     *
     * @returns The sizes array
     */
    const size_t *get() const
    {
        return _sizes;
    }
};

static const NDRange NullRange;
static const NDRange Range_128_1 = NDRange(128, 1);
} // namespace gles

/** Check if the OpenGL ES 3.1 API is available at runtime.
 *
 *  @returns true if the OpenGL ES 3.1 API is available.
 */
bool opengles31_is_available();
} // namespace arm_compute

#endif /* __ARM_COMPUTE_OPENGLES_H__ */
