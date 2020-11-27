/*
 * Copyright (c) 2016-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_NETHRESHOLDKERNEL_H
#define ARM_COMPUTE_NETHRESHOLDKERNEL_H

#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Types.h"
#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the thresholding kernel */
class NEThresholdKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEThresholdKernel";
    }
    /** Default constructor */
    NEThresholdKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEThresholdKernel(const NEThresholdKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEThresholdKernel &operator=(const NEThresholdKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEThresholdKernel(NEThresholdKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEThresholdKernel &operator=(NEThresholdKernel &&) = default;
    /** Default destructor */
    ~NEThresholdKernel() = default;
    /** Initialise the kernel's input, output and threshold parameters.
     *
     * @param[in]  input  An input tensor. Data type supported: U8
     * @param[out] output The output tensor. Data type supported: U8.
     * @param[in]  info   Threshold kernel descriptor
     */
    void configure(const ITensor *input, ITensor *output, const ThresholdKernelInfo &info);
    /** Static function to check if given info will lead to a valid configuration of @ref NEThresholdKernel
     *
     * @param[in] input  Input tensor info. Data type supported: U8
     * @param[in] output Output tensor info. Data type supported: U8
     * @param[in] info   Threshold kernel descriptor
     *
     * @return A status containing an error code in case of failure
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const ThresholdKernelInfo &info);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** run binary thresholding on the given window */
    void run_binary(const Window &window);
    /** run range thresholding on the given window */
    void run_range(const Window &window);

    void (NEThresholdKernel::*_func)(const Window &window);

    const ITensor      *_input;  /**< Input */
    ITensor            *_output; /**< Output */
    ThresholdKernelInfo _info;   /**< Threshold descriptor */
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NETHRESHOLDKERNEL_H */
