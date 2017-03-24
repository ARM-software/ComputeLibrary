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
#ifndef __ARM_COMPUTE_IHOG_H__
#define __ARM_COMPUTE_IHOG_H__

#include "arm_compute/core/Types.h"

#include <cstddef>

namespace arm_compute
{
class HOGInfo;
/** Interface for HOG data-object */
class IHOG
{
public:
    /** Interface to be implemented by the child class to return the HOG's metadata
     *
     * @return A pointer to the HOG's metadata.
     */
    virtual const HOGInfo *info() const = 0;
    /** Default virtual destructor */
    virtual ~IHOG() = default;
    /** Pointer to the first element of the array which stores the linear SVM coefficients of HOG descriptor
     *
     * @note Other elements of the array can be accessed using descriptor()[idx] for idx=[0, descriptor_size() - 1]
     *
     * @return A pointer to the first element of the array which stores the linear SVM coefficients of HOG descriptor
     */
    virtual float *descriptor() const = 0;
};
}
#endif /* __ARM_COMPUTE_IHOG_H__ */
