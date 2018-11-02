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
#ifndef __ARM_COMPUTE_CLPERMUTEKERNEL_H__
#define __ARM_COMPUTE_CLPERMUTEKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** OpenCL kernel to perform tensor permutation.
 *
 * Permutes given a permutation vector
 */
class CLPermuteKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLPermuteKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLPermuteKernel(const CLPermuteKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLPermuteKernel &operator=(const CLPermuteKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLPermuteKernel(CLPermuteKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLPermuteKernel &operator=(CLPermuteKernel &&) = default;
    /** Set the input and output of the kernel.
     *
     * @param[in] input  The input tensor to permute. Data types supported: U8/S8/QS8/QASYMM8/U16/S16/QS16/F16/U32/S32/F32
     * @param[in] output The output tensor. Data types supported: Same as @p input
     * @param[in] perm   Permutation vector
     */
    void configure(const ICLTensor *input, ICLTensor *output, const PermutationVector &perm);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    ICLTensor        *_output;
    PermutationVector _perm;
};
} // arm_compute
#endif /*__ARM_COMPUTE_CLPERMUTEKERNEL_H__ */
