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
#ifndef __ARM_COMPUTE_IGCKERNEL_H__
#define __ARM_COMPUTE_IGCKERNEL_H__

#include "arm_compute/core/GLES_COMPUTE/GCKernelLibrary.h"
#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"

#include "arm_compute/core/IKernel.h"

namespace arm_compute
{
class IGCTensor;
class Window;

/** Common interface for all the GLES kernels */
class IGCKernel : public IKernel
{
public:
    /** Constructor */
    IGCKernel();
    /** Returns a reference to the GLES kernel of this object.
     *
     * @return A reference to the GLES kernel of this object.
     */
    GCKernel &kernel();

    /** Add the passed 1D tensor's parameters to the object's kernel's arguments starting from the index idx.
     *
     * @param[in] idx           Index at which to start adding the tensor's arguments.Input and output tensor will have sperated index, multiple indices start from 1, single index have to be set to 0.
     * @param[in] tensor        Tensor to set as an argument of the object's kernel.
     * @param[in] binding_point Tensor's binding point in this kernel.
     * @param[in] window        Window the kernel will be executed on.
     */
    void add_1D_tensor_argument(unsigned int &idx, const IGCTensor *tensor, const unsigned int binding_point, const Window &window);

    /** Add the passed 2D tensor's parameters to the object's kernel's arguments starting from the index idx.
     *
     * @param[in] idx           Index at which to start adding the tensor's arguments.Input and output tensor will have sperated index, multiple indices start from 1, single index have to be set to 0.
     * @param[in] tensor        Tensor to set as an argument of the object's kernel.
     * @param[in] binding_point Tensor's binding point in this kernel.
     * @param[in] window        Window the kernel will be executed on.
     */
    void add_2D_tensor_argument(unsigned int &idx, const IGCTensor *tensor, const unsigned int binding_point, const Window &window);

    /** Add the passed 3D tensor's parameters to the object's kernel's arguments starting from the index idx.
     *
     * @param[in] idx           Index at which to start adding the tensor's arguments.Input and output tensor will have sperated index, multiple indices start from 1, single index have to be set to 0.
     * @param[in] tensor        Tensor to set as an argument of the object's kernel.
     * @param[in] binding_point Tensor's binding point in this kernel.
     * @param[in] window        Window the kernel will be executed on.
     */
    void add_3D_tensor_argument(unsigned int &idx, const IGCTensor *tensor, const unsigned int binding_point, const Window &window);

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
    /** Enqueue the OpenGL ES shader to process the given window
     *
     * @param[in] window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     */
    virtual void run(const Window &window) = 0;

    /** Set the Local-Workgroup-Size hint
     *
     * @note This method should be called after the configuration of the kernel
     *
     * @param[in] lws_hint Local-Workgroup-Size to use
     */
    void set_lws_hint(gles::NDRange &lws_hint)
    {
        _lws_hint = lws_hint;
    }

private:
    /** Add the passed tensor's parameters to the object's kernel's arguments starting from the index idx.
     *
     * @param[in] idx           Index at which to start adding the tensor's arguments.Input and output tensor will have sperated index, multiple indices start from 1, single index have to be set to 0.
     * @param[in] tensor        Tensor to set as an argument of the object's kernel.
     * @param[in] binding_point Tensor's binding point in this kernel.
     * @param[in] window        Window the kernel will be executed on.
     */
    template <unsigned int dimension_size>
    void add_tensor_argument(unsigned int &idx, const IGCTensor *tensor, const unsigned int binding_point, const Window &window);

    /** Returns the number of arguments enqueued per tensor object.
     *
     * @return The number of arguments enqueued per tensor object.
     */
    template <unsigned int dimension_size>
    unsigned int           num_arguments_per_tensor() const;

protected:
    GCKernel      _kernel;   /**< GLES kernel to run */
    gles::NDRange _lws_hint; /**< Local workgroup size hint for the GLES kernel */
};

/** Add the kernel to the command queue with the given window.
 *
 * @note Depending on the size of the window, this might translate into several jobs being enqueued.
 *
 * @note If kernel->kernel() is empty then the function will return without adding anything to the queue.
 *
 * @param[in] kernel Kernel to enqueue
 * @param[in] window Window the kernel has to process.
 * @param[in] lws    Local workgroup size requested, by default (1, 1, 1)
 *
 * @note If any dimension of the lws is greater than the global workgroup size then no lws will be passed.
 */
void enqueue(IGCKernel &kernel, const Window &window, const gles::NDRange &lws = gles::NDRange(1U, 1U, 1U));
}
#endif /*__ARM_COMPUTE_IGCKERNEL_H__ */
