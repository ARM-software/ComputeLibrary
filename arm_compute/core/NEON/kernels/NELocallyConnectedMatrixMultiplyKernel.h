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
#ifndef __ARM_COMPUTE_NELOCALLYCONNECTEDMATRIXMULTIPLYKERNEL_H__
#define __ARM_COMPUTE_NELOCALLYCONNECTEDMATRIXMULTIPLYKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** NEON kernel to multiply each row of first tensor with low 2 dimensions of second tensor. */
class NELocallyConnectedMatrixMultiplyKernel : public INEKernel
{
public:
    /** Default constructor */
    NELocallyConnectedMatrixMultiplyKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NELocallyConnectedMatrixMultiplyKernel(const NELocallyConnectedMatrixMultiplyKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NELocallyConnectedMatrixMultiplyKernel &operator=(const NELocallyConnectedMatrixMultiplyKernel &) = delete;
    /** Allow instances of this class to be moved */
    NELocallyConnectedMatrixMultiplyKernel(NELocallyConnectedMatrixMultiplyKernel &&) = default;
    /** Allow instances of this class to be moved */
    NELocallyConnectedMatrixMultiplyKernel &operator=(NELocallyConnectedMatrixMultiplyKernel &&) = default;
    /** Initialise the kernel's input and output
     *
     * @param[in]  input0 First input tensor. Data types supported: F16, F32
     * @param[in]  input1 Second input tensor containing the Matrix B. Data type supported: same as @p input0
     * @param[out] output Output tensor to store the result of matrix multiplication. Data type supported: same as @p input0
     */
    void configure(const ITensor *input0, const ITensor *input1, ITensor *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    const ITensor *_input0;
    const ITensor *_input1;
    ITensor       *_output;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NELOCALLYCONNECTEDMATRIXMULTIPLYKERNEL_H__ */
