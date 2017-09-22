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
#ifndef __ARM_COMPUTE_NEBITWISENOTKERNEL_H__
#define __ARM_COMPUTE_NEBITWISENOTKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the kernel to perform bitwise NOT operation
 *
 * Result is computed by:
 * @f[ output(x,y) = \lnot input(x,y) @f]
 */
class NEBitwiseNotKernel : public INEKernel
{
public:
    /** Default constructor */
    NEBitwiseNotKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEBitwiseNotKernel(const NEBitwiseNotKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEBitwiseNotKernel &operator=(const NEBitwiseNotKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEBitwiseNotKernel(NEBitwiseNotKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEBitwiseNotKernel &operator=(NEBitwiseNotKernel &&) = default;
    /** Initialise the kernel's input and output
     *
     * @param[in]  input  An input tensor. Data type supported: U8.
     * @param[out] output The output tensor. Data type supported: U8.
     */
    void configure(const ITensor *input, ITensor *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    const ITensor *_input;  /**< Source tensor */
    ITensor       *_output; /**< Destination tensor */
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEBITWISENOTKERNEL_H__ */
