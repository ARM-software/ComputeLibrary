/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CL_TUNINGPARAMETERS_LIST_H
#define ARM_COMPUTE_CL_TUNINGPARAMETERS_LIST_H

#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/runtime/CL/CLTunerTypes.h"
#include "arm_compute/runtime/CL/CLTuningParams.h"
#include "support/ToolchainSupport.h"

#include <memory>

namespace arm_compute
{
namespace cl_tuner
{
/** Interface for Tuning Parameters lists
 *
 * The tuning parameter lists contain a set of tuning parameters to estimate.
 * There are 3 tuner modes, each using its specific list:
 *  - Exhaustive tuner mode is the slowest during the tuning but will find faster tuning parameters
 *  - Normal tuner mode is the average modality in terms of tuning time and tuning parameters found
 *  - Rapid tuner mode is the fastest but the tuning parameters might not be the fastest
 *
 */
class ICLTuningParametersList
{
public:
    /** Constructor */
    ICLTuningParametersList() = default;
    /** Copy Constructor */
    ICLTuningParametersList(const ICLTuningParametersList &) = default;
    /** Move Constructor */
    ICLTuningParametersList(ICLTuningParametersList &&) noexcept(true) = default;
    /** Assignment */
    ICLTuningParametersList &operator=(const ICLTuningParametersList &) = default;
    /** Move Assignment */
    ICLTuningParametersList &operator=(ICLTuningParametersList &&) noexcept(true) = default;
    /** Destructor */
    virtual ~ICLTuningParametersList() = default;

    /** Return the tuning parameter values at the given index.
     *
     * @return tuning parameter values at the given index
     */
    virtual CLTuningParams operator[](size_t) = 0;

    /** Tuning parameters list size.
     *
     * @return Tuning parameters list size
     */
    virtual size_t size() = 0;
};

/** Construct an ICLTuningParametersList object for the given tuner mode and gws configuration.
 *
 * @param[in] tuning_info Tuning info containng which parameters to tune and the tuner mode
 * @param[in] gws         Global worksize values
 *
 * @return unique_ptr to the requested ICLTuningParametersList implementation.
 */
std::unique_ptr<ICLTuningParametersList> get_tuning_parameters_list(CLTuningInfo tuning_info, const cl::NDRange &gws);

} // namespace cl_tuner
} // namespace arm_compute
#endif /*ARM_COMPUTE_CL_TUNINGPARAMETERS_LIST_H */
