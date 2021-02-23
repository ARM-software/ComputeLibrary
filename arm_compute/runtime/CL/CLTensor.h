/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLTENSOR_H
#define ARM_COMPUTE_CLTENSOR_H

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"

#include <cstdint>

namespace arm_compute
{
// Forward declarations
class ITensorAllocator;
class ITensorInfo;
class IRuntimeContext;
class CLRuntimeContext;
/** Basic implementation of the OpenCL tensor interface */
class CLTensor : public ICLTensor, public IMemoryManageable
{
public:
    /** Constructor.
     *
     * @param[in] ctx (Optional)  Pointer to a @ref CLRuntimeContext.
     *                            If nullptr is passed in, the legacy api using the singletons will be used. Otherwise the memory for the
     *                            tensor will allocate on the context passed in.
     *                            The singletons legacy api has been deprecated and will be removed in future releases.
     */
    CLTensor(IRuntimeContext *ctx = nullptr);

    /** Destructor */
    ~CLTensor() = default;
    /** Prevent copying by construction */
    CLTensor(const CLTensor &) = delete;
    /** Default move constructor */
    CLTensor(CLTensor &&) = default;
    /** Prevent copaingy by assignment */
    CLTensor &operator=(const CLTensor &) = delete;
    /** Default move assignment operator */
    CLTensor &operator=(CLTensor &&) = default;

    /** Return a pointer to the tensor's allocator
     *
     * @return A pointer to the tensor's allocator
     */
    CLTensorAllocator *allocator();
    /** Enqueue a map operation of the allocated buffer.
     *
     * @param[in] blocking If true, then the mapping will be ready to use by the time
     *                     this method returns, else it is the caller's responsibility
     *                     to flush the queue and wait for the mapping operation to have completed.
     */
    void map(bool blocking = true);
    using ICLTensor::map;
    /** Enqueue an unmap operation of the allocated and mapped buffer.
     *
     * @note This method simply enqueues the unmap operation, it is the caller's responsibility to flush the queue and make sure the unmap is finished before
     *       the memory is accessed by the device.
     */
    void unmap();
    using ICLTensor::unmap;

    // Inherited methods overridden:
    TensorInfo       *info() const override;
    TensorInfo       *info() override;
    const cl::Buffer &cl_buffer() const override;
    CLQuantization    quantization() const override;
    void associate_memory_group(IMemoryGroup *memory_group) override;
    CLRuntimeContext *context();

protected:
    // Inherited methods overridden:
    uint8_t *do_map(cl::CommandQueue &q, bool blocking) override;
    void do_unmap(cl::CommandQueue &q) override;

private:
    mutable CLTensorAllocator _allocator; /**< Instance of the OpenCL tensor allocator */
    CLRuntimeContext         *_ctx{ nullptr };
};

/** OpenCL Image */
using CLImage = CLTensor;
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLTENSOR_H */
