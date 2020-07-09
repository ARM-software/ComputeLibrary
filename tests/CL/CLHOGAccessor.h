/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_CLHOGACCESSOR_H
#define ARM_COMPUTE_TEST_CLHOGACCESSOR_H

#include "arm_compute/runtime/CL/CLHOG.h"
#include "tests/IHOGAccessor.h"

namespace arm_compute
{
namespace test
{
/** Accessor implementation for @ref CLHOG objects. */
class CLHOGAccessor : public IHOGAccessor
{
public:
    /** Create an accessor for the given @p CLHOG. */
    CLHOGAccessor(CLHOG &hog)
        : _hog{ hog }
    {
        _hog.map();
    }

    /** Destructor that unmaps the CL memory. */
    ~CLHOGAccessor()
    {
        _hog.unmap();
    }

    /** Prevent instances of this class from being copied (As this class contains references). */
    CLHOGAccessor(const CLHOGAccessor &) = delete;
    /** Prevent instances of this class from being copied (As this class contains references). */
    CLHOGAccessor &operator=(const CLHOGAccessor &) = delete;

    /** Pointer to the first element of the array which stores the linear SVM coefficients of HOG descriptor
     *
     * @return A pointer to the first element of the array which stores the linear SVM coefficients of HOG descriptor
     */
    float *descriptor() const override
    {
        return _hog.descriptor();
    }

private:
    CLHOG &_hog;
};
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_CLHOGACCESSOR_H */
