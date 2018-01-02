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
#ifndef __ARM_COMPUTE_GCGEMMMATRIXACCUMULATEBIASESKERNEL_H__
#define __ARM_COMPUTE_GCGEMMMATRIXACCUMULATEBIASESKERNEL_H__

#include "arm_compute/core/GLES_COMPUTE/IGCKernel.h"

namespace arm_compute
{
/** Interface to add a bias to each row of the input tensor
 *
 */
class GCGEMMMatrixAccumulateBiasesKernel : public IGCKernel
{
public:
    /** Default constructor */
    GCGEMMMatrixAccumulateBiasesKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCGEMMMatrixAccumulateBiasesKernel(const GCGEMMMatrixAccumulateBiasesKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCGEMMMatrixAccumulateBiasesKernel &operator=(const GCGEMMMatrixAccumulateBiasesKernel &) = delete;
    /** Allow instances of this class to be moved */
    GCGEMMMatrixAccumulateBiasesKernel(GCGEMMMatrixAccumulateBiasesKernel &&) = default;
    /** Allow instances of this class to be moved */
    GCGEMMMatrixAccumulateBiasesKernel &operator=(GCGEMMMatrixAccumulateBiasesKernel &&) = default;
    /** Set the accumulate buffer and the biases of the kernel.
     *
     * @param[in, out] accum  The accumulate tensor to convert. Data types supported: F16/F32
     * @param[in]      biases The shared biases tensor to append. It must be 1D tensor. Data types supported: Same as @p input
     */
    void configure(IGCTensor *accum, const IGCTensor *biases);

    // Inherited methods overridden:
    void run(const Window &window) override;

private:
    IGCTensor       *_accum;
    const IGCTensor *_biases;
    gles::NDRange    _lws;
};
}

#endif /*__ARM_COMPUTE_GCGEMMMATRIXACCUMULATEBIASESKERNEL_H__ */
