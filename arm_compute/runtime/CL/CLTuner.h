/*
 * Copyright (c) 2017 ARM Limited.
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

private:
    /** Find optimal LWS using brute-force approach
     *
     * @param[in] kernel OpenCL kernel to be tuned with LWS
     *
     * @return The optimal LWS to use
     */
    cl::NDRange find_optimal_lws(ICLKernel &kernel);

    std::unordered_map<std::string, cl::NDRange> _lws_table;
};
}
#endif /*__ARM_COMPUTE_CLTUNER_H__ */
