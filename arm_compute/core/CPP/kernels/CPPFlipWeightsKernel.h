/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_CPP_FLIP_WEIGHTS_KERNEL_H__
#define __ARM_COMPUTE_CPP_FLIP_WEIGHTS_KERNEL_H__

#include "arm_compute/core/CPP/ICPPKernel.h"

namespace arm_compute
{
class ITensor;

/** CPP kernel to perform 180 degrees flipping on deconvolution weights. */
class CPPFlipWeightsKernel : public ICPPKernel
{
public:
    const char *name() const override
    {
        return "CPPFlipWeightsKernel";
    }
    /** Default constructor */
    CPPFlipWeightsKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CPPFlipWeightsKernel(const CPPFlipWeightsKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CPPFlipWeightsKernel &operator=(const CPPFlipWeightsKernel &) = delete;
    /** Allow instances of this class to be moved */
    CPPFlipWeightsKernel(CPPFlipWeightsKernel &&) = default;
    /** Allow instances of this class to be moved */
    CPPFlipWeightsKernel &operator=(CPPFlipWeightsKernel &&) = default;
    /** Default destructor */
    ~CPPFlipWeightsKernel() = default;

    /** Set the input and output of the kernel.
     *
     * @param[in]  input  The input tensor to flip. Data types supported: QASYMM8/F16/F32
     * @param[out] output The output tensor. Data types supported: Same as @p input
     */
    void configure(const ITensor *input, ITensor *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

    /** Function to perform flipping.
     *
     * @param[in] window_input Input region on which to execute the kernel.
     * @param[in] window       Output region on which to execute the kernel.
     */
    template <typename T>
    void flip_weights(const Window &window_input, const Window &window);

    /** Common signature for all the specialised Flip functions
     *
     * @param[in] window_input Input region on which to execute the kernel.
     * @param[in] window       Output region on which to execute the kernel.
     */
    using FlipWeightsFunction = void (CPPFlipWeightsKernel::*)(const Window &window_input, const Window &window);

private:
    const ITensor      *_input;
    ITensor            *_output;
    FlipWeightsFunction _func;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CPP_FLIP_WEIGHTS_KERNEL_H__ */
