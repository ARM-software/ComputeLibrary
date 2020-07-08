/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CPPTOPKVERNEL_H
#define ARM_COMPUTE_CPPTOPKVERNEL_H

#include "arm_compute/core/CPP/ICPPKernel.h"

namespace arm_compute
{
class ITensor;

/** CPP kernel to perform tensor TopKV operation. */
class CPPTopKVKernel : public ICPPKernel
{
public:
    const char *name() const override
    {
        return "CPPTopKVKernel";
    }
    /** Default constructor */
    CPPTopKVKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CPPTopKVKernel(const CPPTopKVKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CPPTopKVKernel &operator=(const CPPTopKVKernel &) = delete;
    /** Allow instances of this class to be moved */
    CPPTopKVKernel(CPPTopKVKernel &&) = default;
    /** Allow instances of this class to be moved */
    CPPTopKVKernel &operator=(CPPTopKVKernel &&) = default;
    /** Default destructor */
    ~CPPTopKVKernel() = default;

    /** Set the input and output of the kernel.
     *
     * @param[in]  predictions A batch_size x classes tensor. Data types supported: F16/S32/F32/QASYMM8/QASYMM8_SIGNED
     * @param[in]  targets     A batch_size 1D tensor of class ids. Data types supported: U32
     * @param[out] output      Computed precision at @p k as a bool 1D tensor. Data types supported: U8
     * @param[in]  k           Number of top elements to look at for computing precision.
     */
    void configure(const ITensor *predictions, const ITensor *targets, ITensor *output, const unsigned int k);

    /** Static function to check if given info will lead to a valid configuration of @ref CPPTopKVKernel
     *
     * @param[in] predictions A batch_size x classes tensor info. Data types supported: F16/S32/F32/QASYMM8/QASYMM8_SIGNED
     * @param[in] targets     A batch_size 1D tensor info of class ids. Data types supported: U32
     * @param[in] output      Computed precision at @p k as a bool 1D tensor info. Data types supported: U8
     * @param[in] k           Number of top elements to look at for computing precision.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *predictions, const ITensorInfo *targets, ITensorInfo *output, const unsigned int k);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    bool is_parallelisable() const override;

private:
    /** Template function to run the topKV operation. */
    template <typename T>
    void run_topkv();

    const ITensor *_predictions;
    const ITensor *_targets;
    ITensor       *_output;

    unsigned int _k;
    unsigned int _batch_size;
    unsigned int _num_classes;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CPPTOPKVKERNEL_H */
