/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NETHRESHOLD_H
#define ARM_COMPUTE_NETHRESHOLD_H

#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/INESimpleFunctionNoBorder.h"

#include <cstdint>

namespace arm_compute
{
// Forward declarations
class ITensor;
class ITensorInfo;

/** Basic function to run @ref NEThresholdKernel
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
*/
class NEThreshold : public INESimpleFunctionNoBorder
{
public:
    /** Initialise the function's source, destination, thresholds and threshold type
     *
     * @param[in]  input  First tensor input. Data type supported: U8.
     * @param[out] output Output tensor. Data type supported: U8.
     * @param[in]  info   Threshold descriptor
     */
    void configure(const ITensor *input, ITensor *output, const ThresholdKernelInfo &info);
    /** Static function to check if given info will lead to a valid configuration of @ref NEThreshold
     *
     * @param[in] input  First tensor input. Data type supported: U8.
     * @param[in] output Output tensor. Data type supported: U8.
     * @param[in] info   Threshold descriptor.
     *
     * @return A status, containing an error code in case of failure
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const ThresholdKernelInfo &info);
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NETHRESHOLD_H */
