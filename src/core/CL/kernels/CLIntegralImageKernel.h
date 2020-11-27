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
#ifndef ARM_COMPUTE_CLINTEGRALIMAGEKERNEL_H
#define ARM_COMPUTE_CLINTEGRALIMAGEKERNEL_H

#include "src/core/CL/ICLKernel.h"
#include "src/core/CL/ICLSimple2DKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface to run the horizontal pass of the integral image kernel. */
class CLIntegralImageHorKernel : public ICLSimple2DKernel
{
public:
    /** Initialise the kernel's input and output.
     *
     * @param[in]  input  An input tensor. Data types supported: U8
     * @param[out] output Destination tensor, Data types supported: U32.
     */
    void configure(const ICLTensor *input, ICLTensor *output);
    /** Initialise the kernel's input and output.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           An input tensor. Data types supported: U8
     * @param[out] output          Destination tensor, Data types supported: U32.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output);
};

/** Interface to run the vertical pass of the integral image kernel. */
class CLIntegralImageVertKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLIntegralImageVertKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLIntegralImageVertKernel(const CLIntegralImageVertKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLIntegralImageVertKernel &operator=(const CLIntegralImageVertKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLIntegralImageVertKernel(CLIntegralImageVertKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLIntegralImageVertKernel &operator=(CLIntegralImageVertKernel &&) = default;
    /** Initialise the kernel's input and output.
     *
     * @param[in,out] in_out The input/output tensor. Data types supported: U32
     */
    void configure(ICLTensor *in_out);
    /** Initialise the kernel's input and output.
     *
     * @param[in]     compile_context The compile context to be used.
     * @param[in,out] in_out          The input/output tensor. Data types supported: U32
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *in_out);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    ICLTensor *_in_out;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLINTEGRALIMAGEKERNEL_H */
