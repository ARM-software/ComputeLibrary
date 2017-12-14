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

#ifndef __ARM_COMPUTE_GCTENSOR_H__
#define __ARM_COMPUTE_GCTENSOR_H__

#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCTensorAllocator.h"

namespace arm_compute
{
class ITensorAllocator;
class ITensorInfo;

/** Interface for OpenGL ES tensor */
class GCTensor : public IGCTensor
{
public:
    /** Default constructor */
    GCTensor();

    /** Prevent instances of this class from being copied (As this class contains pointers). */
    GCTensor(const GCTensor &) = delete;

    /** Prevent instances of this class from being copy assigned (As this class contains pointers). */
    GCTensor &operator=(const GCTensor &) = delete;

    /** Allow instances of this class to be moved */
    GCTensor(GCTensor &&) = default;

    /** Allow instances of this class to be moved */
    GCTensor &operator=(GCTensor &&) = default;

    /** Virtual destructor */
    virtual ~GCTensor() = default;

    /** Return a pointer to the tensor's allocator
     *
     * @return A pointer to the tensor's allocator
     */
    ITensorAllocator *allocator();

    /** Enqueue a map operation of the allocated buffer on the given queue.
     *
     * @param[in] blocking (Optional) If true, then the mapping will be ready to use by the time
     *                     this method returns, else it is the caller's responsibility
     *                     to flush the queue and wait for the mapping operation to have completed before using the returned mapping pointer.
     *
     * @return The mapping address.
     */
    void map(bool blocking = true);

    /** Enqueue an unmap operation of the allocated and mapped buffer on the given queue.
     *
     * @note This method simply enqueues the unmap operation, it is the caller's responsibility to flush the queue and make sure the unmap is finished before
     *       the memory is accessed by the device.
     *
     */
    void unmap();

    // Inherited methods overridden:
    TensorInfo *info() const override;
    TensorInfo *info() override;
    uint8_t    *buffer() const override;
    GLuint      gc_buffer() const override;

protected:
    // Inherited methods overridden:
    uint8_t *do_map(bool blocking) override;
    void do_unmap() override;

private:
    mutable GCTensorAllocator _allocator;
};

using GCImage = GCTensor;
}

#endif /*__ARM_COMPUTE_GCTENSOR_H__ */
