/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_CPPPERMUTEKERNEL_H__
#define __ARM_COMPUTE_CPPPERMUTEKERNEL_H__

#include "arm_compute/core/CPP/ICPPKernel.h"

namespace arm_compute
{
class ITensor;

/** CPP kernel to perform tensor permutation.
 *
 * Permutes given a permutation vector
 */
class CPPPermuteKernel : public ICPPKernel
{
public:
    /** Default constructor */
    CPPPermuteKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CPPPermuteKernel(const CPPPermuteKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CPPPermuteKernel &operator=(const CPPPermuteKernel &) = delete;
    /** Allow instances of this class to be moved */
    CPPPermuteKernel(CPPPermuteKernel &&) = default;
    /** Allow instances of this class to be moved */
    CPPPermuteKernel &operator=(CPPPermuteKernel &&) = default;
    /** Default destructor */
    ~CPPPermuteKernel() = default;

    /** Set the input and output of the kernel.
     *
     * @param[in]  input  The input tensor to permute. Data types supported: U8/S8/QS8/QASYMM8/U16/S16/QS16/F16/U32/S32/F32
     * @param[out] output The output tensor. Data types supported: Same as @p input
     * @param[in]  perm   Permutation vector
     */
    void configure(const ITensor *input, ITensor *output, const PermutationVector &perm);
    /** Static function to check if given info will lead to a valid configuration of @ref CPPPermuteKernel
     *
     * @param[in] input  The input tensor to permute. Data types supported: U8/S8/QS8/QASYMM8/U16/S16/QS16/F16/U32/S32/F32
     * @param[in] output The output tensor. Data types supported: Same as @p input
     * @param[in] perm   Permutation vector
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const PermutationVector &perm);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Template function to run the permute
     *
     * @param[in] window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     */
    template <typename T>
    void run_permute(const Window &window);

    /** Common signature for all the specialised permute functions
     *
     * @param[in] window Region on which to execute the kernel.
     */
    using PermuteFunctionPtr = void (CPPPermuteKernel::*)(const Window &window);

    PermuteFunctionPtr _func;
    const ITensor     *_input;
    ITensor           *_output;
    PermutationVector  _perm;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CPPPERMUTEKERNEL_H__ */
