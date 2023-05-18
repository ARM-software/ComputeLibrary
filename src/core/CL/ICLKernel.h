/*
 * Copyright (c) 2016-2023 Arm Limited.
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
#ifndef ARM_COMPUTE_ICLKERNEL_H
#define ARM_COMPUTE_ICLKERNEL_H

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLTypes.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/core/IKernel.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/experimental/Types.h"
#include "arm_compute/runtime/CL/CLTuningParams.h"

#include "src/core/CL/DefaultLWSHeuristics.h"

#include <string>

namespace arm_compute
{
namespace
{
bool is_same_lws(cl::NDRange lws0, cl::NDRange lws1)
{
    if(lws0.dimensions() != lws1.dimensions())
    {
        return false;
    }

    for(size_t i = 0; i < lws0.dimensions(); ++i)
    {
        if(lws0.get()[i] != lws1.get()[i])
        {
            return false;
        }
    }

    return true;
}
} // namespace
template <typename T>
class ICLArray;
class ICLTensor;
class Window;
/** Common interface for all the OpenCL kernels */
class ICLKernel : public IKernel
{
private:
    /** Returns the number of arguments enqueued per array object.
     *
     * @return The number of arguments enqueued per array object.
     */
    template <unsigned int        dimension_size>
    constexpr static unsigned int num_arguments_per_array()
    {
        return num_arguments_per_tensor<dimension_size>();
    }
    /** Returns the number of arguments enqueued per tensor object.
     *
     * @return The number of arguments enqueued per tensor object.
     */
    template <unsigned int        dimension_size>
    constexpr static unsigned int num_arguments_per_tensor()
    {
        return 2 + 2 * dimension_size;
    }

    /** Get default lws for the kernel
     *
     * @param[in] window               Execution window used by the kernel
     * @param[in] use_dummy_work_items If the kernel uses dummy workloads
     *
     * @return    cl::NDRange
     */
    cl::NDRange default_lws_tune(const Window &window, bool use_dummy_work_items)
    {
        return get_default_lws_for_type(_type, gws_from_window(window, use_dummy_work_items));
    }

    using IKernel::configure; //Prevent children from calling IKernel::configure() directly
protected:
    /** Configure the kernel's window and local workgroup size hint.
     *
     * @param[in] window    The maximum window which will be returned by window()
     * @param[in] lws_hint  Local-Workgroup-Size to use.
     * @param[in] wbsm_hint (Optional) Workgroup-Batch-Size-Modifier to use.
     */
    void configure_internal(const Window &window, cl::NDRange lws_hint, cl_int wbsm_hint = 0)
    {
        configure_internal(window, CLTuningParams(lws_hint, wbsm_hint));
    }

    /** Configure the kernel's window and tuning parameters hints.
     *
     * @param[in] window             The maximum window which will be returned by window()
     * @param[in] tuning_params_hint (Optional) Tuning parameters to use.
     */
    void configure_internal(const Window &window, CLTuningParams tuning_params_hint = CLTuningParams(CLKernelLibrary::get().default_ndrange(), 0))
    {
        _tuning_params_hint = tuning_params_hint;

        if(is_same_lws(_tuning_params_hint.get_lws(), CLKernelLibrary::get().default_ndrange()))
        {
            // Disable use_dummy_work_items at configure time. Because dummy work items only affect gws size, which
            // will be recalculated with use_dummy_work_items flag at run time again anyway.
            _tuning_params_hint.set_lws(default_lws_tune(window, false /* use_dummy_work_items */));
        }

        IKernel::configure(window);
    }

public:
    /** Constructor */
    ICLKernel()
        : _kernel(nullptr), _target(GPUTarget::MIDGARD), _config_id(arm_compute::default_config_id), _max_workgroup_size(0), _type(CLKernelType::UNKNOWN), _tuning_params_hint(), _cached_gws(cl::NullRange)
    {
    }
    /** Returns a reference to the OpenCL kernel of this object.
     *
     * @return A reference to the OpenCL kernel of this object.
     */
    cl::Kernel &kernel()
    {
        return _kernel;
    }
    /** Returns the CL kernel type
     *
     * @return The CL kernel type
     */
    CLKernelType type() const
    {
        return _type;
    }
    /** Add the passed 1D array's parameters to the object's kernel's arguments starting from the index idx.
     *
     * @param[in,out] idx            Index at which to start adding the array's arguments. Will be incremented by the number of kernel arguments set.
     * @param[in]     array          Array to set as an argument of the object's kernel.
     * @param[in]     strides        @ref Strides object containing stride of each dimension in bytes.
     * @param[in]     num_dimensions Number of dimensions of the @p array.
     * @param[in]     window         Window the kernel will be executed on.
     */
    template <typename T>
    void add_1D_array_argument(unsigned int &idx, const ICLArray<T> *array, const Strides &strides, unsigned int num_dimensions, const Window &window)
    {
        add_array_argument<T, 1>(idx, array, strides, num_dimensions, window);
    }
    /** Add the passed 1D tensor's parameters to the object's kernel's arguments starting from the index idx.
     *
     * @param[in,out] idx    Index at which to start adding the tensor's arguments. Will be incremented by the number of kernel arguments set.
     * @param[in]     tensor Tensor to set as an argument of the object's kernel.
     * @param[in]     window Window the kernel will be executed on.
     */
    void add_1D_tensor_argument(unsigned int &idx, const ICLTensor *tensor, const Window &window)
    {
        add_tensor_argument<1>(idx, tensor, window);
    }
    /** Add the passed 1D tensor's parameters to the object's kernel's arguments starting from the index idx if the condition is true.
     *
     * @param[in]     cond   Condition to check
     * @param[in,out] idx    Index at which to start adding the tensor's arguments. Will be incremented by the number of kernel arguments set.
     * @param[in]     tensor Tensor to set as an argument of the object's kernel.
     * @param[in]     window Window the kernel will be executed on.
     */
    void add_1D_tensor_argument_if(bool cond, unsigned int &idx, const ICLTensor *tensor, const Window &window)
    {
        if(cond)
        {
            add_1D_tensor_argument(idx, tensor, window);
        }
    }
    /** Add the passed 2D tensor's parameters to the object's kernel's arguments starting from the index idx.
     *
     * @param[in,out] idx    Index at which to start adding the tensor's arguments. Will be incremented by the number of kernel arguments set.
     * @param[in]     tensor Tensor to set as an argument of the object's kernel.
     * @param[in]     window Window the kernel will be executed on.
     */
    void add_2D_tensor_argument(unsigned int &idx, const ICLTensor *tensor, const Window &window)
    {
        add_tensor_argument<2>(idx, tensor, window);
    }
    /** Add the passed 2D tensor's parameters to the object's kernel's arguments starting from the index idx if the condition is true.
     *
     * @param[in]     cond   Condition to check
     * @param[in,out] idx    Index at which to start adding the tensor's arguments. Will be incremented by the number of kernel arguments set.
     * @param[in]     tensor Tensor to set as an argument of the object's kernel.
     * @param[in]     window Window the kernel will be executed on.
     */
    void add_2D_tensor_argument_if(bool cond, unsigned int &idx, const ICLTensor *tensor, const Window &window)
    {
        if(cond)
        {
            add_2D_tensor_argument(idx, tensor, window);
        }
    }
    /** Add the passed 3D tensor's parameters to the object's kernel's arguments starting from the index idx.
     *
     * @param[in,out] idx    Index at which to start adding the tensor's arguments. Will be incremented by the number of kernel arguments set.
     * @param[in]     tensor Tensor to set as an argument of the object's kernel.
     * @param[in]     window Window the kernel will be executed on.
     */
    void add_3D_tensor_argument(unsigned int &idx, const ICLTensor *tensor, const Window &window)
    {
        add_tensor_argument<3>(idx, tensor, window);
    }
    /** Add the passed 4D tensor's parameters to the object's kernel's arguments starting from the index idx.
     *
     * @param[in,out] idx    Index at which to start adding the tensor's arguments. Will be incremented by the number of kernel arguments set.
     * @param[in]     tensor Tensor to set as an argument of the object's kernel.
     * @param[in]     window Window the kernel will be executed on.
     */
    void add_4D_tensor_argument(unsigned int &idx, const ICLTensor *tensor, const Window &window)
    {
        add_tensor_argument<4>(idx, tensor, window);
    }
    /** Add the passed 5D tensor's parameters to the object's kernel's arguments starting from the index idx.
     *
     * @param[in,out] idx    Index at which to start adding the tensor's arguments. Will be incremented by the number of kernel arguments set.
     * @param[in]     tensor Tensor to set as an argument of the object's kernel.
     * @param[in]     window Window the kernel will be executed on.
     */
    void add_5D_tensor_argument(unsigned int &idx, const ICLTensor *tensor, const Window &window)
    {
        add_tensor_argument<5>(idx, tensor, window);
    }

    /** Add the passed NHW 3D tensor's parameters to the object's kernel's arguments by passing strides, dimensions and the offset to the first valid element in bytes.
     *
     * @param[in,out] idx    Index at which to start adding the tensor's arguments. Will be incremented by the number of kernel arguments set.
     * @param[in]     tensor Tensor to set as an argument of the object's kernel.
     */
    void add_3d_tensor_nhw_argument(unsigned int &idx, const ICLTensor *tensor);

    /** Returns the number of arguments enqueued per NHW 3D Tensor object.
     *
     * @return The number of arguments enqueued per NHW 3D Tensor object.
     */
    constexpr static unsigned int num_arguments_per_3d_tensor_nhw()
    {
        constexpr unsigned int no_args_per_3d_tensor_nhw = 7u;
        return no_args_per_3d_tensor_nhw;
    }

    /** Add the passed NHWC 4D tensor's parameters to the object's kernel's arguments by passing strides, dimensions and the offset to the first valid element in bytes.
     *
     * @param[in,out] idx    Index at which to start adding the tensor's arguments. Will be incremented by the number of kernel arguments set.
     * @param[in]     tensor Tensor to set as an argument of the object's kernel.
     */
    void add_4d_tensor_nhwc_argument(unsigned int &idx, const ICLTensor *tensor);

    /** Returns the number of arguments enqueued per NHWC 4D Tensor object.
     *
     * @return The number of arguments enqueued per NHWC 4D Tensor object.
     */
    constexpr static unsigned int num_arguments_per_4d_tensor_nhwc()
    {
        constexpr unsigned int no_args_per_4d_tensor_nhwc = 9u;
        return no_args_per_4d_tensor_nhwc;
    }

    /** Returns the number of arguments enqueued per 1D array object.
     *
     * @return The number of arguments enqueues per 1D array object.
     */
    constexpr static unsigned int num_arguments_per_1D_array()
    {
        return num_arguments_per_array<1>();
    }
    /** Returns the number of arguments enqueued per 1D tensor object.
     *
     * @return The number of arguments enqueues per 1D tensor object.
     */
    constexpr static unsigned int num_arguments_per_1D_tensor()
    {
        return num_arguments_per_tensor<1>();
    }
    /** Returns the number of arguments enqueued per 2D tensor object.
     *
     * @return The number of arguments enqueues per 2D tensor object.
     */
    constexpr static unsigned int num_arguments_per_2D_tensor()
    {
        return num_arguments_per_tensor<2>();
    }
    /** Returns the number of arguments enqueued per 3D tensor object.
     *
     * @return The number of arguments enqueues per 3D tensor object.
     */
    constexpr static unsigned int num_arguments_per_3D_tensor()
    {
        return num_arguments_per_tensor<3>();
    }
    /** Returns the number of arguments enqueued per 4D tensor object.
     *
     * @return The number of arguments enqueues per 4D tensor object.
     */
    constexpr static unsigned int num_arguments_per_4D_tensor()
    {
        return num_arguments_per_tensor<4>();
    }
    /** Enqueue the OpenCL kernel to process the given window  on the passed OpenCL command queue.
     *
     * @note The queue is *not* flushed by this method, and therefore the kernel will not have been executed by the time this method returns.
     *
     * @param[in]     window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     * @param[in,out] queue  Command queue on which to enqueue the kernel.
     */
    virtual void run(const Window &window, cl::CommandQueue &queue)
    {
        ARM_COMPUTE_UNUSED(window, queue);
    }
    /** Enqueue the OpenCL kernel to process the given window  on the passed OpenCL command queue.
     *
     * @note The queue is *not* flushed by this method, and therefore the kernel will not have been executed by the time this method returns.
     *
     * @param[in]     tensors A vector containing the tensors to operato on.
     * @param[in]     window  Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     * @param[in,out] queue   Command queue on which to enqueue the kernel.
     */
    virtual void run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
    {
        ARM_COMPUTE_UNUSED(tensors, window, queue);
    }
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
    void set_lws_hint(const cl::NDRange &lws_hint)
    {
        ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this); // lws_hint will be overwritten by configure()
        _tuning_params_hint.set_lws(lws_hint);
    }

    /** Return the Local-Workgroup-Size hint
     *
     * @return Current lws hint
     */
    cl::NDRange lws_hint() const
    {
        return _tuning_params_hint.get_lws();
    }

    /** Set the workgroup batch size modifier hint
     *
     * @note This method should be called after the configuration of the kernel
     *
     * @param[in] wbsm_hint workgroup batch size modifier value
     */
    void set_wbsm_hint(const cl_int &wbsm_hint)
    {
        ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this); // wbsm_hint will be overwritten by configure()
        _tuning_params_hint.set_wbsm(wbsm_hint);
    }

    /** Return the workgroup batch size modifier hint
     *
     * @return Current wbsm hint
     */
    cl_int wbsm_hint() const
    {
        return _tuning_params_hint.get_wbsm();
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
    void set_target(GPUTarget target)
    {
        _target = target;
    }

    /** Set the targeted GPU architecture according to the CL device
     *
     * @param[in] device A CL device
     */
    void set_target(cl::Device &device);

    /** Get the targeted GPU architecture
     *
     * @return The targeted GPU architecture.
     */
    GPUTarget get_target() const
    {
        return _target;
    }

    /** Get the maximum workgroup size for the device the CLKernelLibrary uses.
     *
     * @return The maximum workgroup size value.
     */
    size_t get_max_workgroup_size();
    /** Get the global work size given an execution window
     *
     * @param[in] window               Execution window
     * @param[in] use_dummy_work_items If the kernel uses dummy work items
     *
     * @return Global work size of the given execution window
     */
    static cl::NDRange gws_from_window(const Window &window, bool use_dummy_work_items);

    /** Get the cached gws used to enqueue this kernel
     *
     * @return Latest global work size of the kernel
     */
    cl::NDRange get_cached_gws() const;

    /** Cache the latest gws used to enqueue this kernel
     *
     * @param[in] gws Latest global work size of the kernel
     */
    void cache_gws(const cl::NDRange &gws);

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

protected:
    cl::Kernel   _kernel;             /**< OpenCL kernel to run */
    GPUTarget    _target;             /**< The targeted GPU */
    std::string  _config_id;          /**< Configuration ID */
    size_t       _max_workgroup_size; /**< The maximum workgroup size for this kernel */
    CLKernelType _type;               /**< The CL kernel type */
private:
    CLTuningParams _tuning_params_hint; /**< Tuning parameters hint for the OpenCL kernel */
    cl::NDRange    _cached_gws;         /**< Latest GWS used to enqueue this kernel */
};

/** Add the kernel to the command queue with the given window.
 *
 * @note Depending on the size of the window, this might translate into several jobs being enqueued.
 *
 * @note If kernel->kernel() is empty then the function will return without adding anything to the queue.
 *
 * @param[in,out] queue                OpenCL command queue.
 * @param[in]     kernel               Kernel to enqueue
 * @param[in]     window               Window the kernel has to process.
 * @param[in]     lws_hint             (Optional) Local workgroup size requested. Default is based on the device target.
 * @param[in]     use_dummy_work_items (Optional) Use dummy work items in order to have two dimensional power of two NDRange. Default is false
 *                                     Note: it is kernel responsibility to check if the work-item is out-of-range
 *
 * @note If any dimension of the lws is greater than the global workgroup size then no lws will be passed.
 */
void enqueue(cl::CommandQueue &queue, ICLKernel &kernel, const Window &window, const cl::NDRange &lws_hint = CLKernelLibrary::get().default_ndrange(), bool use_dummy_work_items = false);

/** Add the passed array's parameters to the object's kernel's arguments starting from the index idx.
 *
 * @param[in,out] idx            Index at which to start adding the array's arguments. Will be incremented by the number of kernel arguments set.
 * @param[in]     array          Array to set as an argument of the object's kernel.
 * @param[in]     strides        @ref Strides object containing stride of each dimension in bytes.
 * @param[in]     num_dimensions Number of dimensions of the @p array.
 * @param[in]     window         Window the kernel will be executed on.
 */
template <typename T, unsigned int dimension_size>
void ICLKernel::add_array_argument(unsigned &idx, const ICLArray<T> *array, const Strides &strides, unsigned int num_dimensions, const Window &window)
{
    ARM_COMPUTE_ERROR_ON(array == nullptr);

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

    ARM_COMPUTE_ERROR_ON_MSG_VAR(idx_start + num_arguments_per_array<dimension_size>() != idx,
                                 "add_%dD_array_argument() is supposed to add exactly %d arguments to the kernel", dimension_size, num_arguments_per_array<dimension_size>());
    ARM_COMPUTE_UNUSED(idx_start);
}
}
#endif /*ARM_COMPUTE_ICLKERNEL_H */
