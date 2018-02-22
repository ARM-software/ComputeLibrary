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
#ifndef __ARM_COMPUTE_CLTUNER_H__
#define __ARM_COMPUTE_CLTUNER_H__

#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/runtime/CL/ICLTuner.h"

#include <unordered_map>

namespace arm_compute
{
class ICLKernel;

/** Basic implementation of the OpenCL tuner interface */
class CLTuner : public ICLTuner
{
public:
    /** Constructor */
    CLTuner();

    /** Destructor */
    ~CLTuner() = default;

    /** Import LWS table
     *
     * @param[in] lws_table The unordered_map container to import
     */
    void import_lws_table(const std::unordered_map<std::string, cl::NDRange> &lws_table);

    /** Export LWS table
     *
     * return The lws table as unordered_map container
     */
    const std::unordered_map<std::string, cl::NDRange> &export_lws_table();

    // Inherited methods overridden:
    void tune_kernel(ICLKernel &kernel) override;

    /** Set the OpenCL kernel event
     *
     * @note The interceptor can use this function to store the event associated to the OpenCL kernel
     *
     * @param[in] kernel_event The OpenCL kernel event
     */
    void set_cl_kernel_event(cl_event kernel_event);

    std::function<decltype(clEnqueueNDRangeKernel)> real_function;

private:
    /** Find optimal LWS using brute-force approach
     *
     * @param[in] kernel OpenCL kernel to be tuned with LWS
     *
     * @return The optimal LWS to use
     */
    cl::NDRange find_optimal_lws(ICLKernel &kernel);

    std::unordered_map<std::string, cl::NDRange> _lws_table;
    cl::CommandQueue _queue;
    cl::CommandQueue _queue_profiler;
    cl::Event        _kernel_event;
};

/* Function to be used to intercept kernel enqueues and store their OpenCL Event */
class Interceptor
{
public:
    explicit Interceptor(CLTuner &tuner);

    /** clEnqueueNDRangeKernel interface
     *
     * @param[in] command_queue           A valid command-queue. The kernel will be queued for execution on the device associated with command_queue.
     * @param[in] kernel                  A valid kernel object. The OpenCL context associated with kernel and command_queue must be the same.
     * @param[in] work_dim                The number of dimensions used to specify the global work-items and work-items in the work-group. work_dim must be greater than zero and less than or equal to CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS.
     * @param[in] gwo                     Global-Workgroup-Offset. It can be used to specify an array of work_dim unsigned values that describe the offset used to calculate the global ID of a work-item. If global_work_offset is NULL, the global IDs start at offset (0, 0, ... 0).
     * @param[in] gws                     Global-Workgroup-Size. Points to an array of work_dim unsigned values that describe the number of global work-items in work_dim dimensions that will execute the kernel function.
     * @param[in] lws                     Local-Workgroup-Size. Points to an array of work_dim unsigned values that describe the number of work-items that make up a work-group
     * @param[in] num_events_in_wait_list Number of events in the waiting list
     * @param[in] event_wait_list         Event waiting list
     * @param[in] event                   OpenCL kernel event
     *
     * @return the OpenCL status
     */
    cl_int operator()(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t *gwo, const size_t *gws, const size_t *lws, cl_uint num_events_in_wait_list,
                      const cl_event *event_wait_list, cl_event *event);

private:
    CLTuner &_tuner;
};
}
#endif /*__ARM_COMPUTE_CLTUNER_H__ */
