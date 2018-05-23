/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_RUNTIME_IMEMORY_REGION_H__
#define __ARM_COMPUTE_RUNTIME_IMEMORY_REGION_H__

#include <cstddef>

namespace arm_compute
{
/** Memory region interface */
class IMemoryRegion
{
public:
    /** Default constructor
     *
     * @param[in] size Region size
     */
    IMemoryRegion(size_t size)
        : _size(size)
    {
    }
    /** Virtual Destructor */
    virtual ~IMemoryRegion() = default;
    /** Returns the pointer to the allocated data.
     *
     * @return Pointer to the allocated data
     */
    virtual void *buffer() = 0;
    /** Returns the pointer to the allocated data.
     *
     * @return Pointer to the allocated data
     */
    virtual void *buffer() const = 0;
    /** Handle of internal memory
     *
     * @return Handle of memory
     */
    virtual void **handle() = 0;
    /** Memory region size accessor
     *
     * @return Memory region size
     */
    size_t size()
    {
        return _size;
    }
    /** Sets size of region
     *
     * @warning This should only be used in correlation with handle
     *
     * @param[in] size Size to set
     */
    void set_size(size_t size)
    {
        _size = size;
    }

protected:
    size_t _size;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_RUNTIME_IMEMORY_REGION_H__ */
