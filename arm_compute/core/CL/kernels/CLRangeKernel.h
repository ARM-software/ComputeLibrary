/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLRANGEKERNEL_H__
#define __ARM_COMPUTE_CLRANGEKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ICLTensor;

/** Kernel class for Range
 *
 * range generates a 1-D tensor containing a sequence of numbers that begins at 'start' and extends by increments
 * of 'step' up to but not including 'end'.
 */
class CLRangeKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLRangeKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLRangeKernel(const CLRangeKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLRangeKernel &operator=(const CLRangeKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLRangeKernel(CLRangeKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLRangeKernel &operator=(CLRangeKernel &&) = default;
    /** Default destructor */
    ~CLRangeKernel() = default;
    /** Initialize the kernel's output tensor, start, end and step of the sequence.
     *
     * @param[out] output Output tensor. Data types supported: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32.
     * @param[in]  start  The starting value of the sequence.
     * @param[in]  end    The ending (not including) value of the sequence.
     * @param[in]  step   The gap between each pair of values in the sequence.
     */
    void configure(ICLTensor *output, float start, float end, float step);
    /** Static function to check if given info will lead to a valid configuration of @ref CLRangeKernel
     *
     * @param[in] output Output tensor info. Data types supported: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32.
     * @param[in] start  The starting value of the sequence.
     * @param[in] end    The ending (not including) value of the sequence.
     * @param[in] step   The gap between each pair of values in the sequence.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *output, float start, float end, float step);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    float      _start;  /**< Start of sequence */
    float      _end;    /**< End of sequence */
    float      _step;   /**< Increment/step value */
    ICLTensor *_output; /**< Destination tensor */
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLRANGEKERNEL_H__ */
