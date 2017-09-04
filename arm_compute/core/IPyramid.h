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
#ifndef __ARM_COMPUTE_IPYRAMID_H__
#define __ARM_COMPUTE_IPYRAMID_H__

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/PyramidInfo.h"
#include "arm_compute/core/Types.h"

#include <cstddef>

namespace arm_compute
{
/** Interface for pyramid data-object */
class IPyramid
{
public:
    /** Default virtual destructor */
    virtual ~IPyramid() = default;
    /** Interface to be implemented by the child class to return the Pyramid's metadata
     *
     * @return A pointer to the Pyramid's metadata.
     */
    virtual const PyramidInfo *info() const = 0;
    /** Retrieves a level of the pyramid as a ITensor pointer
     *
     * @param[in] index The index of the level, such that index is less than levels.
     *
     *  @return An ITensor pointer
     */
    virtual ITensor *get_pyramid_level(size_t index) const = 0;
};
}

#endif /* __ARM_COMPUTE_IPYRAMID_H__ */
