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
#ifndef __ARM_COMPUTE_CLLUT_H__
#define __ARM_COMPUTE_CLLUT_H__

#include "arm_compute/core/CL/ICLLut.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLLutAllocator.h"

#include <cstddef>
#include <cstdint>

namespace arm_compute
{
class ILutAllocator;

/** Basic implementation of the OpenCL lut interface */
class CLLut : public ICLLut
{
public:
    /** Constructor */
    CLLut();
    /** Constructor: initializes a LUT which can contain num_values values of data_type type.
     *
     * @param[in] num_elements Number of elements of the LUT.
     * @param[in] data_type    Data type of each element.
     */
    CLLut(size_t num_elements, DataType data_type);
    /** Return a pointer to the lut's allocator
     *
     * @return A pointer to the lut's allocator
     */
    ILutAllocator *allocator();
    /** Enqueue a map operation of the allocated buffer.
     *
     * @param[in] blocking If true, then the mapping will be ready to use by the time
     *                     this method returns, else it is the caller's responsibility
     *                     to flush the queue and wait for the mapping operation to have completed.
     */
    void map(bool blocking = true);
    using ICLLut::map;
    /** Enqueue an unmap operation of the allocated and mapped buffer.
     *
     * @note This method simply enqueues the unmap operation, it is the caller's responsibility to flush the queue and make sure the unmap is finished before
     *       the memory is accessed by the device.
     */
    void unmap();
    using ICLLut::unmap;

    // Inherited methods overridden:
    size_t            num_elements() const override;
    uint32_t          index_offset() const override;
    size_t            size_in_bytes() const override;
    DataType          type() const override;
    const cl::Buffer &cl_buffer() const override;
    void              clear() override;

protected:
    // Inherited methods overridden:
    uint8_t *do_map(cl::CommandQueue &q, bool blocking) override;
    void do_unmap(cl::CommandQueue &q) override;

private:
    CLLutAllocator _allocator; /**< Instance of the OpenCL lut allocator */
};
}
#endif /*__ARM_COMPUTE_CLLUT_H__ */
