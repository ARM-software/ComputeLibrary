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
#ifndef __ARM_COMPUTE_LUT_H__
#define __ARM_COMPUTE_LUT_H__

#include "arm_compute/core/ILut.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/LutAllocator.h"

#include <cstddef>
#include <cstdint>

namespace arm_compute
{
class ILutAllocator;

/** Basic implementation of the LUT interface */
class Lut : public ILut
{
public:
    /** Constructor */
    Lut();
    /** Constructor: initializes a LUT which can contain num_values values of data_type type.
     *
     * @param[in] num_elements Number of elements of the LUT.
     * @param[in] data_type    Data type of each element.
     */
    Lut(size_t num_elements, DataType data_type);
    /** Return a pointer to the lut's allocator
     *
     * @return A pointer to the lut's allocator
     */
    ILutAllocator *allocator();

    // Inherited methods overridden:
    size_t   num_elements() const override;
    uint32_t index_offset() const override;
    size_t   size_in_bytes() const override;
    DataType type() const override;
    uint8_t *buffer() const override;
    void     clear() override;

private:
    LutAllocator _allocator; /**< Instance of the basic CPU allocator.*/
};
}
#endif /* __ARM_COMPUTE_LUT_H__ */
