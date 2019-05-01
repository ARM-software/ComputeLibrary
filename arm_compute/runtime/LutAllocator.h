/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_LUTALLOCATOR_H__
#define __ARM_COMPUTE_LUTALLOCATOR_H__

#include "arm_compute/runtime/ILutAllocator.h"

#include <cstdint>
#include <memory>

namespace arm_compute
{
/** Basic implementation of a CPU memory LUT allocator. */
class LutAllocator : public ILutAllocator
{
public:
    /** Default constructor. */
    LutAllocator();
    /** Interface to be implemented by the child class to return the pointer to the allocate data.
     *
     * @return a pointer to the data.
     */
    uint8_t *data() const;

protected:
    /** Allocate num_elements() * sizeof(type()) of CPU memory. */
    void allocate() override;
    /** No-op for CPU memory
     *
     * @return A pointer to the beginning of the look up table's allocation.
     */
    uint8_t *lock() override;
    /** No-op for CPU memory. */
    void unlock() override;

private:
    mutable std::vector<uint8_t> _buffer; /**< CPU memory allocation. */
};
}
#endif /* __ARM_COMPUTE_LUTALLOCATOR_H__ */
