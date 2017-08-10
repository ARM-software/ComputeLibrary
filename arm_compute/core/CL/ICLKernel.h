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
#ifndef __ARM_COMPUTE_ICLKERNEL_H__
#define __ARM_COMPUTE_ICLKERNEL_H__

#include "arm_compute/core/CL/CLTypes.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/IKernel.h"

namespace arm_compute
{
class ICLTensor;
class Window;

/** Common interface for all the OpenCL kernels */
class ICLKernel : public IKernel
{
public:
    /** Constructor */
    ICLKernel();
    /** Returns a reference to the OpenCL kernel of this object.
     *
     * @return A reference to the OpenCL kernel of this object.
     */
    cl::Kernel &kernel();
    /** Add the passed 1D tensor's parameters to the object's kernel's arguments starting from the index idx.
     *
     * @param[in,out] idx    Index at which to start adding the tensor's arguments. Will be incremented by the number of kernel arguments set.
     * @param[in]     tensor Tensor to set as an argument of the object's kernel.
     * @param[in]     window Window the kernel will be executed on.
     */
    void add_1D_tensor_argument(unsigned int &idx, const ICLTensor *tensor, const Window &window);
    /** Add the passed 2D tensor's parameters to the object's kernel's arguments starting from the index idx.
     *
     * @param[in,out] idx    Index at which to start adding the tensor's arguments. Will be incremented by the number of kernel arguments set.
     * @param[in]     tensor Tensor to set as an argument of the object's kernel.
     * @param[in]     window Window the kernel will be executed on.
     */
    void add_2D_tensor_argument(unsigned int &idx, const ICLTensor *tensor, const Window &window);
    /** Add the passed 3D tensor's parameters to the object's kernel's arguments starting from the index idx.
     *
     * @param[in,out] idx    Index at which to start adding the tensor's arguments. Will be incremented by the number of kernel arguments set.
     * @param[in]     tensor Tensor to set as an argument of the object's kernel.
     * @param[in]     window Window the kernel will be executed on.
     */
    void add_3D_tensor_argument(unsigned int &idx, const ICLTensor *tensor, const Window &window);
    /** Add the passed 4D tensor's parameters to the object's kernel's arguments starting from the index idx.
     *
     * @param[in,out] idx    Index at which to start adding the tensor's arguments. Will be incremented by the number of kernel arguments set.
     * @param[in]     tensor Tensor to set as an argument of the object's kernel.
     * @param[in]     window Window the kernel will be executed on.
     */
    void add_4D_tensor_argument(unsigned int &idx, const ICLTensor *tensor, const Window &window);
    /** Returns the number of arguments enqueued per 1D tensor object.
     *
     * @return The number of arguments enqueues per 1D tensor object.
     */
    unsigned int num_arguments_per_1D_tensor() const;
    /** Returns the number of arguments enqueued per 2D tensor object.
     *
     * @return The number of arguments enqueues per 2D tensor object.
     */
    unsigned int num_arguments_per_2D_tensor() const;
    /** Returns the number of arguments enqueued per 3D tensor object.
     *
     * @return The number of arguments enqueues per 3D tensor object.
     */
    unsigned int num_arguments_per_3D_tensor() const;
    /** Returns the number of arguments enqueued per 4D tensor object.
     *
     * @return The number of arguments enqueues per 4D tensor object.
     */
    unsigned int num_arguments_per_4D_tensor() const;
    /** Enqueue the OpenCL kernel to process the given window  on the passed OpenCL command queue.
     *
     * @note The queue is *not* flushed by this method, and therefore the kernel will not have been executed by the time this method returns.
     *
     * @param[in]     window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     * @param[in,out] queue  Command queue on which to enqueue the kernel.
     */
    virtual void run(const Window &window, cl::CommandQueue &queue) = 0;
    /** Add the passed parameters to the object's kernel's arguments starting from the index idx.
     *
     * @param[in,out] idx   Index at which to start adding the arguments. Will be incremented by the number of kernel arguments set.
     * @param[in]     value Value to set as an argument of the object's kernel.
     */
    template <typename T>
    void add_argument(unsigned int &idx, T value)
    {
        _kernel.setArg(idx++, value);
    }

    /** Set the Local-Workgroup-Size hint
     *
     * @note This method should be called after the configuration of the kernel
     *
     * @param[in] lws_hint Local-Workgroup-Size to use
     */
    void set_lws_hint(cl::NDRange &lws_hint)
    {
        _lws_hint = lws_hint;
    }

    /** Set the targeted GPU architecture
     *
     * @param[in] target The targeted GPU architecture
     */
    void set_target(GPUTarget target);

    /** Set the targeted GPU architecture according to the CL device
     *
     * @param[in] device A CL device
     */
    void set_target(cl::Device &device);

    /** Get the targeted GPU architecture
     *
     * @return The targeted GPU architecture.
     */
    GPUTarget get_target() const;

private:
    /** Add the passed tensor's parameters to the object's kernel's arguments starting from the index idx.
     *
     * @param[in,out] idx    Index at which to start adding the tensor's arguments. Will be incremented by the number of kernel arguments set.
     * @param[in]     tensor Tensor to set as an argument of the object's kernel.
     * @param[in]     window Window the kernel will be executed on.
     */
    template <unsigned int dimension_size>
    void add_tensor_argument(unsigned int &idx, const ICLTensor *tensor, const Window &window);
    /** Returns the number of arguments enqueued per tensor object.
     *
     * @return The number of arguments enqueued per tensor object.
     */
    template <unsigned int dimension_size>
    unsigned int           num_arguments_per_tensor() const;

protected:
    cl::Kernel  _kernel;   /**< OpenCL kernel to run */
    cl::NDRange _lws_hint; /**< Local workgroup size hint for the OpenCL kernel */
    GPUTarget   _target;   /**< The targeted GPU */
};

/** Add the kernel to the command queue with the given window.
 *
 * @note Depending on the size of the window, this might translate into several jobs being enqueued.
 *
 * @note If kernel->kernel() is empty then the function will return without adding anything to the queue.
 *
 * @param[in,out] queue    OpenCL command queue.
 * @param[in]     kernel   Kernel to enqueue
 * @param[in]     window   Window the kernel has to process.
 * @param[in]     lws_hint Local workgroup size requested, by default (128,1)
 *
 * @note If any dimension of the lws is greater than the global workgroup size then no lws will be passed.
 */
void enqueue(cl::CommandQueue &queue, ICLKernel &kernel, const Window &window, const cl::NDRange &lws_hint = cl::Range_128_1);
}
#endif /*__ARM_COMPUTE_ICLKERNEL_H__ */
