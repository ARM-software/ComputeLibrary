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
#ifndef __ARM_COMPUTE_ILUT_H__
#define __ARM_COMPUTE_ILUT_H__

#include "arm_compute/core/Types.h"

#include <cstddef>

namespace arm_compute
{
/** Lookup Table object interface. */
class ILut
{
public:
    /** Default virtual destructor */
    virtual ~ILut() = default;
    /** Returns the total number of elements in the LUT.
     *
     * @return Total number of elements.
     */
    virtual size_t num_elements() const = 0;
    /** Indicates the offset that needs to be applied to the raw index before performing a lookup in the LUT.
     *
     * @return The normalization offset.
     */
    virtual uint32_t index_offset() const = 0;
    /** Returns the total size in bytes of the LUT.
     *
     * @return Total size of the LUT in bytes.
     */
    virtual size_t size_in_bytes() const = 0;
    /** Returns the type of the LUT.
     *
     * @return The type of the LUT.
     */
    virtual DataType type() const = 0;
    /** Returns a pointer to the start of the LUT.
     * Other elements of the LUT can be accessed using buffer()[idx] for 0 <= idx < num_elements().
     *
     * @return Pointer to the start of the lut.
     */
    virtual uint8_t *buffer() const = 0;
    /** Clears the LUT by setting every element to zero. */
    virtual void clear() = 0;
};
}
#endif /* __ARM_COMPUTE_ILUT_H__ */
