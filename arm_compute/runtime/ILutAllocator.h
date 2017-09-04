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
#ifndef __ARM_COMPUTE_ILUTALLOCATOR_H__
#define __ARM_COMPUTE_ILUTALLOCATOR_H__

#include "arm_compute/core/Types.h"

#include <cstddef>
#include <cstdint>

namespace arm_compute
{
/** Basic interface to allocate LUTs' */
class ILutAllocator
{
public:
    /** Default constructor */
    ILutAllocator();
    /** Default virtual destructor */
    virtual ~ILutAllocator() = default;
    /** Allow instances of this class to be move constructed */
    ILutAllocator(ILutAllocator &&) = default;
    /** Allow instances of this class to be moved */
    ILutAllocator &operator=(ILutAllocator &&) = default;
    /** Allocate an LUT of the requested number of elements and data_type.
     *
     * @param[in] num_elements Number of elements of the LUT.
     * @param[in] data_type    Data type of each element.
     */
    void init(size_t num_elements, DataType data_type);
    /** Returns the total number of elements in the LUT.
     *
     * @return Total number of elements.
     */
    size_t num_elements() const;
    /** Returns the type of the LUT.
     *
     * @return The type of the LUT.
     */
    DataType type() const;
    /** Returns the total size in bytes of the LUT.
     *
     * @return Total size of the LUT in bytes.
     */
    size_t size() const;

protected:
    /** Interface to be implemented by the child class to allocate the LUT. */
    virtual void allocate() = 0;
    /** Interface to be implemented by the child class to lock the memory allocation for the CPU to access.
     *
     * @return Pointer to a CPU mapping of the memory
     */
    virtual uint8_t *lock() = 0;
    /** Interface to be implemented by the child class to unlock the memory allocation after the CPU is done accessing it. */
    virtual void unlock() = 0;

private:
    size_t   _num_elements; /**< Number of elements allocated */
    DataType _data_type;    /**< Data type of LUT elements. */
};
}
#endif /* __ARM_COMPUTE_ILUTALLOCATOR_H__ */
