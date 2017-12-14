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

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLTypes.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/IKernel.h"

#include <string>

namespace arm_compute
{
template <typename T>
class ICLArray;
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
    /** Add the passed 1D array's parameters to the object's kernel's arguments starting from the index idx.
     *
     * @param[in,out] idx            Index at which to start adding the array's arguments. Will be incremented by the number of kernel arguments set.
     * @param[in]     array          Array to set as an argument of the object's kernel.
     * @param[in]     strides        @ref Strides object containing stride of each dimension in bytes.
     * @param[in]     num_dimensions Number of dimensions of the @p array.
     * @param[in]     window         Window the kernel will be executed on.
     */
    template <typename T>
    void add_1D_array_argument(unsigned int &idx, const ICLArray<T> *array, const Strides &strides, unsigned int num_dimensions, const Window &window);
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
    /** Returns the number of arguments enqueued per 1D array object.
     *
     * @return The number of arguments enqueues per 1D array object.
     */
    unsigned int num_arguments_per_1D_array() const;
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

    /** Get the configuration ID
     *
     * @note The configuration ID can be used by the caller to distinguish different calls of the same OpenCL kernel
     *       In particular, this method can be used by CLScheduler to keep track of the best LWS for each configuration of the same kernel.
     *       The configuration ID should be provided only for the kernels potentially affected by the LWS geometry
     *
     * @note This method should be called after the configuration of the kernel
     *
     * @return configuration id string
     */
    const std::string &config_id() const
    {
        return _config_id;
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

    /** Get the maximum workgroup size for the device the CLKernelLibrary uses.
     *
     * @return The maximum workgroup size value.
     */
    size_t get_max_workgroup_size();
    /** Get the global work size given an execution window
     *
     * @param[in] window Execution window
     *
     * @return Global work size of the given execution window
     */
    static cl::NDRange gws_from_window(const Window &window);

private:
    /** Add the passed array's parameters to the object's kernel's arguments starting from the index idx.
     *
     * @param[in,out] idx            Index at which to start adding the array's arguments. Will be incremented by the number of kernel arguments set.
     * @param[in]     array          Array to set as an argument of the object's kernel.
     * @param[in]     strides        @ref Strides object containing stride of each dimension in bytes.
     * @param[in]     num_dimensions Number of dimensions of the @p array.
     * @param[in]     window         Window the kernel will be executed on.
     */
    template <typename T, unsigned int dimension_size>
    void add_array_argument(unsigned int &idx, const ICLArray<T> *array, const Strides &strides, unsigned int num_dimensions, const Window &window);
    /** Add the passed tensor's parameters to the object's kernel's arguments starting from the index idx.
     *
     * @param[in,out] idx    Index at which to start adding the tensor's arguments. Will be incremented by the number of kernel arguments set.
     * @param[in]     tensor Tensor to set as an argument of the object's kernel.
     * @param[in]     window Window the kernel will be executed on.
     */
    template <unsigned int dimension_size>
    void add_tensor_argument(unsigned int &idx, const ICLTensor *tensor, const Window &window);
    /** Returns the number of arguments enqueued per array object.
     *
     * @return The number of arguments enqueued per array object.
     */
    template <unsigned int dimension_size>
    unsigned int           num_arguments_per_array() const;
    /** Returns the number of arguments enqueued per tensor object.
     *
     * @return The number of arguments enqueued per tensor object.
     */
    template <unsigned int dimension_size>
    unsigned int           num_arguments_per_tensor() const;

protected:
    cl::Kernel  _kernel;             /**< OpenCL kernel to run */
    cl::NDRange _lws_hint;           /**< Local workgroup size hint for the OpenCL kernel */
    GPUTarget   _target;             /**< The targeted GPU */
    std::string _config_id;          /**< Configuration ID */
    size_t      _max_workgroup_size; /**< The maximum workgroup size for this kernel */
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
 * @param[in]     lws_hint Local workgroup size requested, by default (128,1).
 *
 * @note If any dimension of the lws is greater than the global workgroup size then no lws will be passed.
 */
void enqueue(cl::CommandQueue &queue, ICLKernel &kernel, const Window &window, const cl::NDRange &lws_hint = CLKernelLibrary::get().default_ndrange());

template <typename T, unsigned int dimension_size>
void ICLKernel::add_array_argument(unsigned &idx, const ICLArray<T> *array, const Strides &strides, unsigned int num_dimensions, const Window &window)
{
    // Calculate offset to the start of the window
    unsigned int offset_first_element = 0;

    for(unsigned int n = 0; n < num_dimensions; ++n)
    {
        offset_first_element += window[n].start() * strides[n];
    }

    unsigned int idx_start = idx;
    _kernel.setArg(idx++, array->cl_buffer());

    for(unsigned int dimension = 0; dimension < dimension_size; dimension++)
    {
        _kernel.setArg<cl_uint>(idx++, strides[dimension]);
        _kernel.setArg<cl_uint>(idx++, strides[dimension] * window[dimension].step());
    }

    _kernel.setArg<cl_uint>(idx++, offset_first_element);

    ARM_COMPUTE_ERROR_ON_MSG(idx_start + num_arguments_per_array<dimension_size>() != idx,
                             "add_%dD_array_argument() is supposed to add exactly %d arguments to the kernel", dimension_size, num_arguments_per_array<dimension_size>());
    ARM_COMPUTE_UNUSED(idx_start);
}

template <typename T>
void ICLKernel::add_1D_array_argument(unsigned int &idx, const ICLArray<T> *array, const Strides &strides, unsigned int num_dimensions, const Window &window)
{
    add_array_argument<T, 1>(idx, array, strides, num_dimensions, window);
}

template <unsigned int dimension_size>
unsigned int           ICLKernel::num_arguments_per_array() const
{
    return num_arguments_per_tensor<dimension_size>();
}

template <unsigned int dimension_size>
unsigned int           ICLKernel::num_arguments_per_tensor() const
{
    return 2 + 2 * dimension_size;
}
}
#endif /*__ARM_COMPUTE_ICLKERNEL_H__ */
