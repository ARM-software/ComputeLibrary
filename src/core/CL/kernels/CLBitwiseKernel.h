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
#ifndef ARM_COMPUTE_CLBITWISEKERNEL_H
#define ARM_COMPUTE_CLBITWISEKERNEL_H

#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the bitwise operation kernel.
 *
 * Result depends on the \ref BitwiseOperation and is computed by:
 * AND operation: @f[ output(x,y) = input1(x,y) \land input2(x,y) @f]
 * NOT operation: @f[ output(x,y) = \lnot input1(x,y) @f]
 * OR operation: @f[ output(x,y) = input1(x,y) \lor input2(x,y) @f]
 * XOR operation: @f[ output(x,y) = input1(x,y) \oplus input2(x,y) @f]
 */
class CLBitwiseKernel : public ICLKernel
{
public:
    /** Default constructor. */
    CLBitwiseKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLBitwiseKernel(const CLBitwiseKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLBitwiseKernel &operator=(const CLBitwiseKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLBitwiseKernel(CLBitwiseKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLBitwiseKernel &operator=(CLBitwiseKernel &&) = default;
    /** Set the inputs and output tensors
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input1          Source tensor. Data types supported: U8.
     * @param[in]  input2          Source tensor. Data types supported: U8.
     * @param[out] output          Destination tensor. Data types supported: U8.
     * @param[in]  op              Bitwise operation to perform. Supported: AND, OR, NOT, XOR.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, BitwiseOperation op);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input1; /**< Source tensor 1 */
    const ICLTensor *_input2; /**< Source tensor 2 */
    ICLTensor       *_output; /**< Destination tensor */
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLBITWISEKERNEL_H */
