/*
 * Copyright (c) 2016-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CLSCHEDULER_H
#define ARM_COMPUTE_CLSCHEDULER_H

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLTypes.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/experimental/Types.h"
#include "arm_compute/runtime/CL/CLGEMMHeuristicsHandle.h"
#include "arm_compute/runtime/CL/CLHelpers.h"
#include "arm_compute/runtime/CL/CLTypes.h"
#include "arm_compute/runtime/CL/ICLTuner.h"

namespace arm_compute
{
class ICLKernel;
class ICLTuner;
/** Provides global access to a CL context and command queue. */
class CLScheduler final
{
public:
    /** Constructor */
    CLScheduler();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLScheduler(const CLScheduler &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLScheduler &operator=(const CLScheduler &) = delete;
    /** Default destructor */
    ~CLScheduler() = default;
    /** Access the scheduler singleton.
     * This method has been deprecated and will be removed in future releases
     * @return The scheduler
     */
    static CLScheduler &get();
    /** Initialises the context and command queue used by the scheduler to default values
     *  and sets a default device and kernel path for the @ref CLKernelLibrary.
     *
     * @param[in] cl_tuner        (Optional) Pointer to ICLTuner (default=nullptr)
     * @param[in] gemm_h          (Optional) Pointer to CLGEMMHeuristicsHandle (default = nullptr)
     * @param[in] cl_backend_type (Optional) Type of backend to use (default = CLBackendType::Native)
     */
    void default_init(ICLTuner *cl_tuner = nullptr, CLGEMMHeuristicsHandle *gemm_h = nullptr, CLBackendType cl_backend_type = CLBackendType::Native);
    /** Initialises the scheduler with context and device provided by the user
     *
     * @param[in] device   OpenCL device to be used
     * @param[in] ctx      OpenCL ctx to be used
     * @param[in] cl_tuner (Optional) Pointer to ICLTuner (default=nullptr)
     * @param[in] gemm_h   (Optional) Pointer to CLGEMMHeuristicsHandle (default = nullptr)
     */
    void default_init_with_context(cl::Device &device, cl::Context &ctx, ICLTuner *cl_tuner = nullptr, CLGEMMHeuristicsHandle *gemm_h = nullptr);

    /** Re-initializes the context and command queue used by the scheduler to default values
     *  and sets a default device and kernel path for the @ref CLKernelLibrary.
     *
     * @param[in] cl_tuner        (Optional) Pointer to ICLTuner (default=nullptr)
     * @param[in] gemm_h          (Optional) Pointer to CLGEMMHeuristicsHandle (default = nullptr)
     * @param[in] cl_backend_type (Optional) Type of backend to use (default = CLBackendType::Native)
     */
    void default_reinit(ICLTuner *cl_tuner = nullptr, CLGEMMHeuristicsHandle *gemm_h = nullptr, CLBackendType cl_backend_type = CLBackendType::Native);

    /** Schedule the execution of the passed kernel if possible.
     *
     * @param[in] kernel Kernel to execute.
     * @param[in] flush  (Optional) Specifies if the command queue will be flushed after running the kernel. This will be ignored if job chaining is enabled.
     */
    void enqueue(ICLKernel &kernel, bool flush = true);
    /** Schedule the execution of the passed kernel if possible.
     *
     * @param[in] kernel  Kernel to execute.
     * @param[in] tensors Vector containing the tensors to operate on.
     * @param[in] flush   (Optional) Specifies if the command queue will be flushed after running the kernel. This will be ignored if job chaining is enabled.
     */
    void enqueue_op(ICLKernel &kernel, ITensorPack &tensors, bool flush = true);
    /** Initialises the context and command queue to be used by the scheduler.
     *
     * @param[in] context         A CL context.
     * @param[in] queue           A CL command queue.
     * @param[in] device          A CL device.
     * @param[in] cl_tuner        (Optional) Pointer to OpenCL tuner (default=nullptr)
     *                            Note: It is caller's responsibility to release the allocated memory for CLTuner
     * @param[in] gemm_h          (Optional) Pointer to CLGEMMHeuristicsHandle (default = nullptr)
     * @param[in] cl_backend_type (Optional) Type of backend to use (default = CLBackendType::Native)
     */
    void init(cl::Context context, cl::CommandQueue queue, const cl::Device &device, ICLTuner *cl_tuner = nullptr, CLGEMMHeuristicsHandle *gemm_h = nullptr,
              CLBackendType cl_backend_type = CLBackendType::Native);

    /** Accessor for the associated CL context.
     *
     * @return A CL context.
     */
    cl::Context &context();

    /** Accessor for the associated CL command queue.
     *
     * @return A CL command queue.
     */
    cl::CommandQueue &queue();

    /** Get the target GPU.
     *
     * @return The target GPU.
     */
    GPUTarget target() const;

    /** Accessor for the associated CLGEMMHeuristicsHandle
     *
     * @return Pointer to CLGEMMHeuristicsHandle
     */
    CLGEMMHeuristicsHandle *gemm_heuristics() const;

    /** Accessor to set the CL context to be used by the scheduler.
     *
     * @param[in] context A CL context.
     */
    void set_context(cl::Context context);

    /** Accessor to set the CL command queue to be used by the scheduler.
     *
     * @param[in] queue A CL command queue.
     */
    void set_queue(cl::CommandQueue queue);

    /** Accessor to set target GPU to be used by the scheduler.
     *
     * @param[in] target The target GPU.
     */
    void set_target(GPUTarget target);

    /** Accessor to set the CL tuner to be used by the scheduler.
     *
     * @param[in] tuner A CL tuner
     */
    void set_tuner(ICLTuner *tuner);

    /** Blocks until all commands in the associated command queue have finished. */
    void sync();

    /** Enqueues a marker into the associated command queue and return the event.
     *
     * @return An event that can be waited on to block the executing thread.
     */
    cl::Event enqueue_sync_event();

    /** Tunes OpenCL kernel
     *
     * @param[in] kernel Kernel to tune
     */
    void tune_kernel_static(ICLKernel &kernel);

    /** Enable job chaining. The command queue will only be flushed when @p job_chaining_size kernels have been enqueued.
     *
     * @param[in] job_chaining_size Kernels to enqueue before flushing
     */
    void enable_job_chaining(int job_chaining_size);

    bool is_initialised() const;

private:
    void enqueue_common(ICLKernel &kernel, ITensorPack &tensors, bool flush);
    /** If job chain is disabled, then flush the command queue according to @p flush. Otherwise @p flush is ignored and the queue is only flushed when job chain count exceeds allocated job chain size
     *
     * @param[in] flush Flush the command queue. Ignored when job chain is enabled.
     */
    void flush_queue(bool flush);

    /** Flag to ensure symbols initialisation is happening before Scheduler creation */
    static std::once_flag _initialize_symbols;

    cl::Context             _context;
    cl::CommandQueue        _queue;
    GPUTarget               _target;
    bool                    _is_initialised;
    ICLTuner               *_cl_tuner;
    CLGEMMHeuristicsHandle *_gemm_heuristics;
    CLBackendType           _backend_type;
    bool                    _job_chaining_enabled;
    int                     _job_chaining_size;
    int                     _job_chaining_count;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLSCHEDULER_H */
