/*
 * Copyright (c) 2017-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CLTUNER_H
#define ARM_COMPUTE_CLTUNER_H

#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/utils/misc/Macros.h"
#include "arm_compute/runtime/CL/CLTunerTypes.h"
#include "arm_compute/runtime/CL/CLTuningParams.h"
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
     * @param[in] tuning_info      (Optional) opencl parameters to tune
     *
     */
    CLTuner(bool tune_new_kernels = true, CLTuningInfo tuning_info = CLTuningInfo());

    /** Destructor */
    ~CLTuner() = default;

    /** Setter for tune_new_kernels option
     *
     * @param[in] tune_new_kernels Find the optimal local workgroup size for kernels which are not present in the table ?
     */
    void set_tune_new_kernels(bool tune_new_kernels);

    /** Tune kernels that are not in the tuning parameters table
     *
     * @return True if tuning of new kernels is enabled.
     */
    bool tune_new_kernels() const;

    /** Setter for tune parameters option
     *
     * @param[in] tuning_info opencl parameters to tune
     */
    void set_tuning_parameters(CLTuningInfo tuning_info);

    /** Set OpenCL tuner mode
     *
     * @param[in] mode Indicates how exhaustive the search for the optimal tuning parameters should be while tuning. Default is Exhaustive mode
     */
    void set_tuner_mode(CLTunerMode mode);

    /** Manually add tuning parameters for a kernel
     *
     * @param[in] kernel_id             Unique identifiant of the kernel
     * @param[in] optimal_tuning_params Optimal tuning parameters to use for the given kernel
     */
    void add_tuning_params(const std::string &kernel_id, CLTuningParams optimal_tuning_params);

    /** Import tuning parameters table
     *
     * @param[in] tuning_params_table The unordered_map container to import
     */
    void import_tuning_params(const std::unordered_map<std::string, CLTuningParams> &tuning_params_table);

    /** Give read access to the tuning params table
     *
     * @return The tuning params table as unordered_map container
     */
    const std::unordered_map<std::string, CLTuningParams> &tuning_params_table() const;

    /** Set the OpenCL kernel event
     *
     * @note The interceptor can use this function to store the event associated to the OpenCL kernel
     *
     * @param[in] kernel_event The OpenCL kernel event
     */
    void set_cl_kernel_event(cl_event kernel_event);

    /** clEnqueueNDRangeKernel symbol */
    std::function<decltype(clEnqueueNDRangeKernel)> real_clEnqueueNDRangeKernel;

    /** Load the tuning parameters table from file. It also sets up the tuning read from the file
     *
     * @param[in] filename Load the tuning parameters table from this file.(Must exist)
     *
     */
    void load_from_file(const std::string &filename);

    /** Save the content of the tuning parameters table to file
     *
     * @param[in] filename Save the tuning parameters table to this file. (Content will be overwritten)
     *
     * @return true if the file was created
     */
    bool save_to_file(const std::string &filename) const;

    // Inherited methods overridden:
    void tune_kernel_static(ICLKernel &kernel) override;
    void tune_kernel_dynamic(ICLKernel &kernel) override;
    void tune_kernel_dynamic(ICLKernel &kernel, ITensorPack &tensors) override;
    /** Is the kernel_event set ?
     *
     * @return true if the kernel_event is set.
     */
    bool kernel_event_is_set() const;

    /** A wrapper wrapping tensors and other objects needed for running the kernel
     */
    struct IKernelData;

private:
    /** Perform tune_kernel_dynamic
     *
     * @param[in]     kernel OpenCL kernel to be tuned with tuning parameters
     * @param[in,out] data   IKernelData object wrapping tensors and other objects needed for running the kernel
     *
     */
    void do_tune_kernel_dynamic(ICLKernel &kernel, IKernelData *data);
    /** Find optimal tuning parameters using brute-force approach
     *
     * @param[in]     kernel OpenCL kernel to be tuned with tuning parameters
     * @param[in,out] data   IKernelData object wrapping tensors and other objects needed for running the kernel
     *
     * @return The optimal tuning parameters to use
     */
    CLTuningParams find_optimal_tuning_params(ICLKernel &kernel, IKernelData *data);

    std::unordered_map<std::string, CLTuningParams> _tuning_params_table;
    std::unordered_map<std::string, cl::NDRange>    _lws_table;
    cl::Event    _kernel_event;
    bool         _tune_new_kernels;
    CLTuningInfo _tuning_info;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLTUNER_H */
