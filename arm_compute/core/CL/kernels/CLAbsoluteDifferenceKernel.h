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
#ifndef __ARM_COMPUTE_CLABSOLUTEDIFFERENCEKERNEL_H__
#define __ARM_COMPUTE_CLABSOLUTEDIFFERENCEKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the absolute difference kernel.
 *
 * Absolute difference is computed by:
 * @f[ output(x,y) = | input1(x,y) - input2(x,y) | @f]
 */
class CLAbsoluteDifferenceKernel : public ICLKernel
{
public:
    /** Default constructor. */
    CLAbsoluteDifferenceKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    CLAbsoluteDifferenceKernel(const CLAbsoluteDifferenceKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    CLAbsoluteDifferenceKernel &operator=(const CLAbsoluteDifferenceKernel &) = delete;
    /** Allow instances of this class to be moved. */
    CLAbsoluteDifferenceKernel(CLAbsoluteDifferenceKernel &&) = default;
    /** Allow instances of this class to be moved. */
    CLAbsoluteDifferenceKernel &operator=(CLAbsoluteDifferenceKernel &&) = default;
    /** Default destructor */
    ~CLAbsoluteDifferenceKernel() = default;

    /** Set the inputs and output images.
     *
     * @param[in]  input1 Source tensor. Data types supported: U8/S16.
     * @param[in]  input2 Source tensor. Data types supported: U8/S16.
     * @param[out] output Destination tensor. Data types supported: U8/S16.
     */
    void configure(const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input1; /**< Source tensor 1. */
    const ICLTensor *_input2; /**< Source tensor 2. */
    ICLTensor       *_output; /**< Destination tensor. */
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLABSOLUTEDIFFERENCEKERNEL_H__ */
