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
#ifndef __ARM_COMPUTE_NEGEMMMATRIXACCUMULATEBIASESKERNEL_H__
#define __ARM_COMPUTE_NEGEMMMATRIXACCUMULATEBIASESKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;
/** NEON kernel to add a bias to each row of the input tensor */
class NEGEMMMatrixAccumulateBiasesKernel : public INEKernel
{
public:
    /** Default constructor */
    NEGEMMMatrixAccumulateBiasesKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMMatrixAccumulateBiasesKernel(const NEGEMMMatrixAccumulateBiasesKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMMatrixAccumulateBiasesKernel &operator=(const NEGEMMMatrixAccumulateBiasesKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEGEMMMatrixAccumulateBiasesKernel(NEGEMMMatrixAccumulateBiasesKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEGEMMMatrixAccumulateBiasesKernel &operator=(NEGEMMMatrixAccumulateBiasesKernel &&) = default;
    /** Default destructor */
    ~NEGEMMMatrixAccumulateBiasesKernel() = default;
    /** Set the accumulate buffer and the biases of the kernel.
     *
     * @param[in, out] accum  The accumulate tensor to convert. Data type supported: QS8/QS16/F32
     * @param[in]      biases The shared biases tensor to append. It must be 1D Tensor. Data type supported: Same as @p input
     */
    void configure(ITensor *accum, const ITensor *biases);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    ITensor       *_accum;
    const ITensor *_biases;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEGEMMMATRIXACCUMULATEBIASESKERNEL_H__ */
