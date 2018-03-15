/*
 * Copyright (c) 2017-2018 ARM Limited.
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

#ifndef __ARM_COMPUTE_GCTENSORALLOCATOR_H__
#define __ARM_COMPUTE_GCTENSORALLOCATOR_H__

#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"
#include "arm_compute/runtime/ITensorAllocator.h"
#include "arm_compute/runtime/MemoryGroupBase.h"

#include <memory>

namespace arm_compute
{
class GCTensor;
template <typename>
class MemoryGroupBase;
using GCMemoryGroup = MemoryGroupBase<GCTensor>;

class GLBufferWrapper
{
public:
    GLBufferWrapper()
        : _ssbo_name(0)
    {
        ARM_COMPUTE_GL_CHECK(glGenBuffers(1, &_ssbo_name));
    }
    ~GLBufferWrapper()
    {
        ARM_COMPUTE_GL_CHECK(glDeleteBuffers(1, &_ssbo_name));
    }
    GLuint _ssbo_name;
};
/** Basic implementation of a GLES memory tensor allocator. */
class GCTensorAllocator : public ITensorAllocator
{
public:
    /** Default constructor. */
    GCTensorAllocator(GCTensor *owner = nullptr);

    /** Prevent instances of this class from being copied (As this class contains pointers). */
    GCTensorAllocator(const GCTensorAllocator &) = delete;

    /** Prevent instances of this class from being copy assigned (As this class contains pointers). */
    GCTensorAllocator &operator=(const GCTensorAllocator &) = delete;

    /** Allow instances of this class to be moved */
    GCTensorAllocator(GCTensorAllocator &&) = default;

    /** Allow instances of this class to be moved */
    GCTensorAllocator &operator=(GCTensorAllocator &&) = default;

    /** Default destructor */
    ~GCTensorAllocator();

    /** Interface to be implemented by the child class to return the pointer to the mapped data. */
    uint8_t *data();

    /** Get the OpenGL ES buffer object name
     *
     * @return The buffer object name
     */
    GLuint get_gl_ssbo_name() const;

    /** Enqueue a map operation of the allocated buffer on the given queue.
     *
     * @param[in] blocking If true, then the mapping will be ready to use by the time
     *                     this method returns, else it is the caller's responsibility
     *                     to flush the queue and wait for the mapping operation to have completed before using the returned mapping pointer.
     *
     * @return The mapping address.
     */
    uint8_t *map(bool blocking);

    /** Enqueue an unmap operation of the allocated buffer on the given queue.
     *
     * @note This method simply enqueue the unmap operation, it is the caller's responsibility to flush the queue and make sure the unmap is finished before
     *       the memory is accessed by the device.
     *
     */
    void unmap();

    /** Allocate size specified by TensorInfo of GLES memory.
     *
     * @note: The tensor must not already be allocated when calling this function.
     *
     */
    void allocate() override;

    /** Free allocated GLES memory.
     *
     * @note The tensor must have been allocated when calling this function.
     *
     */
    void free() override;

    /** Associates the tensor with a memory group
     *
     * @param[in] associated_memory_group Memory group to associate the tensor with
     */
    void set_associated_memory_group(GCMemoryGroup *associated_memory_group);

protected:
    /** Call map() on the SSBO.
     *
     * @return A pointer to the beginning of the tensor's allocation.
     */
    uint8_t *lock() override;

    /** Call unmap() on the SSBO. */
    void unlock() override;

private:
    GCMemoryGroup                   *_associated_memory_group; /**< Registered memory group */
    std::unique_ptr<GLBufferWrapper> _gl_buffer;               /**< OpenGL ES object containing the tensor data. */
    uint8_t                         *_mapping;                 /**< Pointer to the CPU mapping of the OpenGL ES buffer. */
    GCTensor                        *_owner;                   /**< Owner of the allocator */
};
}

#endif /* __ARM_COMPUTE_GCTENSORALLOCATOR_H__ */
