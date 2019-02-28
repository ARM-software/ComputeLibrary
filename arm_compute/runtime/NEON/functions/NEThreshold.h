/*
 * Copyright (c) 2016-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_NETHRESHOLD_H__
#define __ARM_COMPUTE_NETHRESHOLD_H__

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/INESimpleFunctionNoBorder.h"

#include <cstdint>

namespace arm_compute
{
class ITensor;

/** Basic function to run @ref NEThresholdKernel */
class NEThreshold : public INESimpleFunctionNoBorder
{
public:
    /** Initialise the function's source, destination, thresholds and threshold type
     *
     * @param[in]  input       First tensor input. Data type supported: U8.
     * @param[out] output      Output tensor. Data type supported: U8.
     * @param[in]  threshold   Threshold. If upper threshold is specified, this will be used as the lower threshold
     * @param[in]  false_value Value to assign when the condition is false
     * @param[in]  true_value  value to assign when the condition is true
     * @param[in]  type        Thresholding type. Can either be BINARY or RANGE.
     * @param[in]  upper       Upper threshold. Only used with RANGE thresholding
     */
    void configure(const ITensor *input, ITensor *output, uint8_t threshold, uint8_t false_value = 0, uint8_t true_value = 0,
                   ThresholdType type = ThresholdType::BINARY, uint8_t upper = 0);
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NETHRESHOLD_H__ */
