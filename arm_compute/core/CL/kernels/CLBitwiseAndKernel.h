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
#ifndef __ARM_COMPUTE_CLBITWISEANDKERNEL_H__
#define __ARM_COMPUTE_CLBITWISEANDKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the bitwise AND operation kernel.
 *
 * Result is computed by:
 * @f[ output(x,y) = input1(x,y) \land input2(x,y) @f]
 */
class CLBitwiseAndKernel : public ICLKernel
{
public:
    /** Default constructor. */
    CLBitwiseAndKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    CLBitwiseAndKernel(const CLBitwiseAndKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    CLBitwiseAndKernel &operator=(const CLBitwiseAndKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLBitwiseAndKernel(CLBitwiseAndKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLBitwiseAndKernel &operator=(CLBitwiseAndKernel &&) = default;
    /** Set the inputs and output images
     *
     * @param[in]  input1 Source tensor. Data types supported: U8.
     * @param[in]  input2 Source tensor. Data types supported: U8.
     * @param[out] output Destination tensor. Data types supported: U8.
     */
    void configure(const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input1; /**< Source tensor 1 */
    const ICLTensor *_input2; /**< Source tensor 2 */
    ICLTensor       *_output; /**< Destination tensor */
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLBITWISEANDKERNEL_H__ */
