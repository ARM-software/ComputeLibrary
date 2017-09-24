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
#ifndef __ARM_COMPUTE_CLARRAY_H__
#define __ARM_COMPUTE_CLARRAY_H__

#include "arm_compute/core/CL/ICLArray.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

namespace arm_compute
{
/** CLArray implementation  */
template <class T>
class CLArray : public ICLArray<T>
{
public:
    /** Default constructor: empty array */
    CLArray()
        : ICLArray<T>(0), _buffer()
    {
    }
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLArray(const CLArray &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLArray &operator=(const CLArray &) = delete;
    CLArray(CLArray &&)                 = default;
    CLArray &operator=(CLArray &&) = default;
    /** Constructor: initializes an array which can contain up to max_num_points values
     *
     * @param[in] max_num_values Maximum number of values the array will be able to stored
     */
    CLArray(size_t max_num_values)
        : ICLArray<T>(max_num_values), _buffer(CLScheduler::get().context(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, max_num_values * sizeof(T))
    {
    }
    /** Enqueue a map operation of the allocated buffer.
     *
     * @param[in] blocking If true, then the mapping will be ready to use by the time
     *                     this method returns, else it is the caller's responsibility
     *                     to flush the queue and wait for the mapping operation to have completed.
     */
    void map(bool blocking = true)
    {
        ICLArray<T>::map(CLScheduler::get().queue(), blocking);
    }
    using ICLArray<T>::map;
    /** Enqueue an unmap operation of the allocated and mapped buffer.
     *
     * @note This method simply enqueues the unmap operation, it is the caller's responsibility to flush the queue and make sure the unmap is finished before
     *       the memory is accessed by the device.
     */
    void unmap()
    {
        ICLArray<T>::unmap(CLScheduler::get().queue());
    }
    using ICLArray<T>::unmap;

    // Inherited methods overridden:
    const cl::Buffer &cl_buffer() const override
    {
        return _buffer;
    }

protected:
    // Inherited methods overridden:
    uint8_t *do_map(cl::CommandQueue &q, bool blocking) override
    {
        ARM_COMPUTE_ERROR_ON(nullptr == _buffer.get());
        return static_cast<uint8_t *>(q.enqueueMapBuffer(_buffer, blocking ? CL_TRUE : CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, 0, this->max_num_values() * sizeof(T)));
    }
    void do_unmap(cl::CommandQueue &q, uint8_t *mapping) override
    {
        ARM_COMPUTE_ERROR_ON(nullptr == _buffer.get());
        q.enqueueUnmapMemObject(_buffer, mapping);
    }

private:
    cl::Buffer _buffer;
};

using CLKeyPointArray        = CLArray<KeyPoint>;
using CLCoordinates2DArray   = CLArray<Coordinates2D>;
using CLDetectionWindowArray = CLArray<DetectionWindow>;
using CLROIArray             = CLArray<ROI>;
using CLSize2DArray          = CLArray<Size2D>;
using CLUInt8Array           = CLArray<cl_uchar>;
using CLUInt16Array          = CLArray<cl_ushort>;
using CLUInt32Array          = CLArray<cl_uint>;
using CLInt16Array           = CLArray<cl_short>;
using CLInt32Array           = CLArray<cl_int>;
using CLFloatArray           = CLArray<cl_float>;
}
#endif /* __ARM_COMPUTE_CLARRAY_H__ */
