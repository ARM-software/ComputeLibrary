/*
 * Copyright (c) 2017-2019 ARM Limited.
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

    /** Setter for tune_new_kernels option
     *
     * @param[in] tune_new_kernels Find the optimal local workgroup size for kernels which are not present in the table ?
     */
    void set_tune_new_kernels(bool tune_new_kernels);
    /** Tune kernels that are not in the LWS table
     *
     * @return True if tuning of new kernels is enabled.
     */
    bool tune_new_kernels() const;
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

    /** Give read access to the LWS table
     *
     * @return The lws table as unordered_map container
     */
    const std::unordered_map<std::string, cl::NDRange> &lws_table() const;

    /** Set the OpenCL kernel event
     *
     * @note The interceptor can use this function to store the event associated to the OpenCL kernel
     *
     * @param[in] kernel_event The OpenCL kernel event
     */
    void set_cl_kernel_event(cl_event kernel_event);

    /** clEnqueueNDRangeKernel symbol */
    std::function<decltype(clEnqueueNDRangeKernel)> real_clEnqueueNDRangeKernel;

    /** Load the LWS table from file
     *
     * @param[in] filename Load the LWS table from this file.(Must exist)
     */
    void load_from_file(const std::string &filename);

    /** Save the content of the LWS table to file
     *
     * @param[in] filename Save the LWS table to this file. (Content will be overwritten)
     */
    void save_to_file(const std::string &filename) const;

    // Inherited methods overridden:
    void tune_kernel_static(ICLKernel &kernel) override;
    void tune_kernel_dynamic(ICLKernel &kernel) override;

    /** Is the kernel_event set ?
     *
     * @return true if the kernel_event is set.
     */
    bool kernel_event_is_set() const;

private:
    /** Find optimal LWS using brute-force approach
     *
     * @param[in] kernel OpenCL kernel to be tuned with LWS
     *
     * @return The optimal LWS to use
     */
    cl::NDRange find_optimal_lws(ICLKernel &kernel);

    std::unordered_map<std::string, cl::NDRange> _lws_table;
    cl::Event _kernel_event;
    bool      _tune_new_kernels;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLTUNER_H__ */
