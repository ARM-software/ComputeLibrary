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
#ifndef __ARM_COMPUTE_ICLARRAY_H__
#define __ARM_COMPUTE_ICLARRAY_H__

#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/IArray.h"
#include "arm_compute/core/ITensor.h"

namespace arm_compute
{
/** Interface for OpenCL Array */
template <class T>
class ICLArray : public IArray<T>
{
public:
    /** Constructor
     *
     * @param[in] max_num_values Maximum size of the array.
     *
     */
    explicit ICLArray(size_t max_num_values)
        : IArray<T>(max_num_values), _mapping(nullptr)
    {
    }

    /** Prevent instances of this class from being copy constructed */
    ICLArray(const ICLArray &) = delete;
    /** Prevent instances of this class from being copied */
    ICLArray &operator=(const ICLArray &) = delete;
    /** Allow instances of this class to be move constructed */
    ICLArray(ICLArray &&) = default;
    /** Allow instances of this class to be moved */
    ICLArray &operator=(ICLArray &&) = default;
    /** Default virtual destructor. */
    virtual ~ICLArray() = default;
    /** Interface to be implemented by the child class to return a reference to the OpenCL buffer containing the array's data.
     *
     * @return A reference to an OpenCL buffer containing the array's data.
     */
    virtual const cl::Buffer &cl_buffer() const = 0;
    /** Enqueue a map operation of the allocated buffer on the given queue.
     *
     * @param[in,out] q        The CL command queue to use for the mapping operation.
     * @param[in]     blocking If true, then the mapping will be ready to use by the time
     *                         this method returns, else it is the caller's responsibility
     *                         to flush the queue and wait for the mapping operation to have completed before using the returned mapping pointer.
     *
     * @return The mapping address.
     */
    void map(cl::CommandQueue &q, bool blocking = true)
    {
        _mapping = do_map(q, blocking);
    }
    /** Enqueue an unmap operation of the allocated and mapped buffer on the given queue.
     *
     * @note This method simply enqueues the unmap operation, it is the caller's responsibility to flush the queue and make sure the unmap is finished before
     *       the memory is accessed by the device.
     *
     * @param[in,out] q The CL command queue to use for the mapping operation.
     */
    void unmap(cl::CommandQueue &q)
    {
        do_unmap(q, _mapping);
        _mapping = nullptr;
    }

    // Inherited methods overridden:
    T *buffer() const override
    {
        return reinterpret_cast<T *>(_mapping);
    }

protected:
    /** Method to be implemented by the child class to map the OpenCL buffer
     *
     * @param[in,out] q        The CL command queue to use for the mapping operation.
     * @param[in]     blocking If true, then the mapping will be ready to use by the time
     *                         this method returns, else it is the caller's responsibility
     *                         to flush the queue and wait for the mapping operation to have completed before using the returned mapping pointer.
     */
    virtual uint8_t *do_map(cl::CommandQueue &q, bool blocking) = 0;
    /** Method to be implemented by the child class to unmap the OpenCL buffer
     *
     * @note This method simply enqueues the unmap operation, it is the caller's responsibility to flush the queue and make sure the unmap is finished before
     *       the memory is accessed by the device.
     *
     * @param[in,out] q       The CL command queue to use for the mapping operation.
     * @param[in]     mapping Pointer to the buffer to be unmapped.
     */
    virtual void do_unmap(cl::CommandQueue &q, uint8_t *mapping) = 0;

private:
    uint8_t *_mapping;
};

/** Interface for OpenCL Array of Key Points. */
using ICLKeyPointArray = ICLArray<KeyPoint>;
/** Interface for OpenCL Array of 2D Coordinates. */
using ICLCoordinates2DArray = ICLArray<Coordinates2D>;
/** Interface for OpenCL Array of Detection Windows. */
using ICLDetectionWindowArray = ICLArray<DetectionWindow>;
/** Interface for OpenCL Array of 2D Sizes. */
using ICLSize2DArray = ICLArray<Size2D>;
/** Interface for OpenCL Array of uint8s. */
using ICLUInt8Array = ICLArray<cl_uchar>;
/** Interface for OpenCL Array of uint16s. */
using ICLUInt16Array = ICLArray<cl_ushort>;
/** Interface for OpenCL Array of uint32s. */
using ICLUInt32Array = ICLArray<cl_uint>;
/** Interface for OpenCL Array of int16s. */
using ICLInt16Array = ICLArray<cl_short>;
/** Interface for OpenCL Array of int32s. */
using ICLInt32Array = ICLArray<cl_int>;
/** Interface for OpenCL Array of floats. */
using ICLFloatArray = ICLArray<cl_float>;
}
#endif /*__ARM_COMPUTE_ICLARRAY_H__*/
