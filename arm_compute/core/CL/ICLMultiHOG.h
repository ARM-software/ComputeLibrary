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
#ifndef __ARM_COMPUTE_ICLMULTIHOG_H__
#define __ARM_COMPUTE_ICLMULTIHOG_H__

#include "arm_compute/core/CL/ICLHOG.h"
#include "arm_compute/core/IMultiHOG.h"

namespace arm_compute
{
/** Interface for storing multiple HOG data-objects */
class ICLMultiHOG : public IMultiHOG
{
public:
    /** Return a pointer to the requested OpenCL HOG model
     *
     * @param[in] index The index of the wanted OpenCL HOG model.
     *
     *  @return A pointer pointed to the HOG model
     */
    virtual ICLHOG *cl_model(size_t index) = 0;
    /** Return a constant pointer to the requested OpenCL HOG model
     *
     * @param[in] index The index of the wanted OpenCL HOG model.
     *
     *  @return A constant pointer pointed to the OpenCL HOG model
     */
    virtual const ICLHOG *cl_model(size_t index) const = 0;

    // Inherited methods overridden:
    IHOG *model(size_t index) override;
    const IHOG *model(size_t index) const override;
};
}
#endif /*__ARM_COMPUTE_ICLMULTIHOG_H__ */
