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
#ifndef __ARM_COMPUTE_NEABSOLUTEDIFFERENCEKERNEL_H__
#define __ARM_COMPUTE_NEABSOLUTEDIFFERENCEKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the absolute difference kernel
 *
 * Absolute difference is computed by:
 * @f[ output(x,y) = | input1(x,y) - input2(x,y) | @f]
 */
class NEAbsoluteDifferenceKernel : public INEKernel
{
public:
    /** Default constructor */
    NEAbsoluteDifferenceKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEAbsoluteDifferenceKernel(const NEAbsoluteDifferenceKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEAbsoluteDifferenceKernel &operator=(const NEAbsoluteDifferenceKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEAbsoluteDifferenceKernel(NEAbsoluteDifferenceKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEAbsoluteDifferenceKernel &operator=(NEAbsoluteDifferenceKernel &&) = default;
    /** Default destructor */
    ~NEAbsoluteDifferenceKernel() = default;

    /** Set the inputs and output tensors
     *
     * @param[in]  input1 Source tensor. Data types supported: U8/S16
     * @param[in]  input2 Source tensor. Data types supported: U8/S16
     * @param[out] output Destination tensor, Data types supported: U8/S16
     */
    void configure(const ITensor *input1, const ITensor *input2, ITensor *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Common signature for all the specialised absolute difference functions
     *
     * @param[in]  input1 An input tensor. Data types supported: U8/S16.
     * @param[in]  input2 An input tensor. Data types supported: U8/S16.
     * @param[out] output The output tensor, Data types supported: U8 (Only if both inputs are U8), S16.
     * @param[in]  window Region on which to execute the kernel.
     */
    using AbsDiffFunction = void(const ITensor *input1, const ITensor *input2, ITensor *output, const Window &window);

    /** Absolute difference function to use for the particular tensor formats passed to configure() */
    AbsDiffFunction *_func;
    const ITensor   *_input1;
    const ITensor   *_input2;
    ITensor         *_output;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEABSOLUTEDIFFERENCEKERNEL_H__ */
