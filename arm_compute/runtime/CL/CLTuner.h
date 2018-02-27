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
    /** Constructor
     *
     * @param[in] tune_new_kernels Find the optimal local workgroup size for kernels which are not present in the table ?
     *
     */
    CLTuner(bool tune_new_kernels = true);

    /** Destructor */
    ~CLTuner() = default;

    /* Setter for tune_new_kernels option
     *
     * @param[in] tune_new_kernels Find the optimal local workgroup size for kernels which are not present in the table ?
     */
    void set_tune_new_kernels(bool tune_new_kernels);
    /** Manually add a LWS for a kernel
     *
     * @param[in] kernel_id   Unique identifiant of the kernel
     * @param[in] optimal_lws Optimal local workgroup size to use for the given kernel
     */
    void add_lws_to_table(const std::string &kernel_id, cl::NDRange optimal_lws);
    /** Import LWS table
     *
     * @param[in] lws_table The unordered_map container to import
     */
    void import_lws_table(const std::unordered_map<std::string, cl::NDRange> &lws_table);

    /** Export LWS table
     *
     * return The lws table as unordered_map container
     */
    const std::unordered_map<std::string, cl::NDRange> &export_lws_table() const;

    /** Set the OpenCL kernel event
     *
     * @note The interceptor can use this function to store the event associated to the OpenCL kernel
     *
     * @param[in] kernel_event The OpenCL kernel event
     */
    void set_cl_kernel_event(cl_event kernel_event);

    std::function<decltype(clEnqueueNDRangeKernel)> real_clEnqueueNDRangeKernel;

    // Inherited methods overridden:
    void tune_kernel(ICLKernel &kernel) override;

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
    bool             _tune_new_kernels;
};

class CLFileTuner : public CLTuner
{
public:
    /** Constructor
     *
     * @param[in] file_path        File to load/store the tuning information from
     * @param[in] update_file      If true, save the new LWS table to the file on exit.
     * @param[in] tune_new_kernels Find the optimal local workgroup size for kernels which are not present in the table ?
     */
    CLFileTuner(std::string file_path = "acl_tuner.csv", bool update_file = false, bool tune_new_kernels = false);

    /** Save the content of the LWS table to file
     */
    void save_to_file() const;
    /* Setter for update_file option
     *
     * @param[in] update_file If true, save the new LWS table to the file on exit.
     */
    void set_update_file(bool update_file);
    /** Destructor
     *
     * Will save the LWS table to the file if the CLFileTuner was created with update_file enabled.
     */
    ~CLFileTuner();
    const std::string filename;

private:
    bool _update_file;
};
}
#endif /*__ARM_COMPUTE_CLTUNER_H__ */
