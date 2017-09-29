/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_NETHRESHOLDKERNEL_H__
#define __ARM_COMPUTE_NETHRESHOLDKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Types.h"

#include <cstdint>

namespace arm_compute
{
class ITensor;

/** Interface for the thresholding kernel
 *
 */
class NEThresholdKernel : public INEKernel
{
public:
    /** Constructor
     * Initialize all the pointers to nullptr and parameters to zero.
     */
    NEThresholdKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEThresholdKernel(const NEThresholdKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEThresholdKernel &operator=(const NEThresholdKernel &) = delete;
    /** Initialise the kernel's input, output and threshold parameters.
     *
     * @param[in]  input       An input tensor. Data type supported: U8
     * @param[out] output      The output tensor. Data type supported: U8.
     * @param[in]  threshold   Threshold. When the threhold type is RANGE, this is used as the lower threshold.
     * @param[in]  false_value value to set when the condition is not respected.
     * @param[in]  true_value  value to set when the condition is respected.
     * @param[in]  type        Thresholding type. Either RANGE or BINARY.
     * @param[in]  upper       Upper threshold. Only used when the thresholding type is RANGE.
     */
    void configure(const ITensor *input, ITensor *output, uint8_t threshold, uint8_t false_value, uint8_t true_value, ThresholdType type, uint8_t upper);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** run binary thresholding on the given window */
    void run_binary(const Window &window);
    /** run range thresholding on the given window */
    void run_range(const Window &window);

    void (NEThresholdKernel::*_func)(const Window &window);

    const ITensor *_input;  /**< Input */
    ITensor       *_output; /**< Output */
    uint8_t        _threshold;
    uint8_t        _false_value;
    uint8_t        _true_value;
    uint8_t        _upper;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NETHRESHOLDKERNEL_H__ */
