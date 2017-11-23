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
#ifndef __ARM_COMPUTE_IMULTIHOG_H__
#define __ARM_COMPUTE_IMULTIHOG_H__

#include "arm_compute/core/IHOG.h"

#include <cstddef>

namespace arm_compute
{
/** Interface for storing multiple HOG data-objects */
class IMultiHOG
{
public:
    /** Default destructor */
    virtual ~IMultiHOG() = default;
    /** The number of HOG models stored
     *
     * @return The number of HOG models stored
     */
    virtual size_t num_models() const = 0;
    /** Return a pointer to the requested HOG model
     *
     * @param[in] index The index of the wanted HOG model.
     *
     *  @return A pointer pointed to the HOG model
     */
    virtual IHOG *model(size_t index) = 0;
    /** Return a const pointer to the requested HOG model
     *
     * @param[in] index The index of the wanted HOG model.
     *
     *  @return A const pointer pointed to the HOG model
     */
    virtual const IHOG *model(size_t index) const = 0;
};
}

#endif /* __ARM_COMPUTE_IMULTIHOG_H__ */
