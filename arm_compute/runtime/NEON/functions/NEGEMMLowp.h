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
#ifndef __ARM_COMPUTE_NEGEMMLOWP_H__
#define __ARM_COMPUTE_NEGEMMLOWP_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/Tensor.h"

#include "arm_compute/core/NEON/kernels/NEFillBorderKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMInterleave4x4Kernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMLowpMatrixMultiplyKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMTranspose1xWKernel.h"

namespace arm_compute
{
class ITensor;

/** Basic function to execute GEMMLowp on NEON. This function calls the following NEON kernels:
*
*  -# @ref NEGEMMInterleave4x4Kernel
*  -# @ref NEGEMMTranspose1xWKernel
*  -# @ref NEGEMMLowpMatrixMultiplyKernel
*
*/
class NEGEMMLowp : public IFunction
{
public:
    /** Constructor */
    NEGEMMLowp();
    /** Initialise the kernel's inputs, output
    *
    * @note GEMM_LOWP:  low precision GEMM kernel
    *  This kernel performs the following computation:
    *
    *  -# Convert a values from uint8 to int32 and add a_offset to each of them.
    *  -# Convert b values from uint8 to int32 and add b_offset to each of them.
    *  -# Compute the int32 matrix product of the resulting a * b.
    *  -# Add output_offset to each entry of the result.
    *  -# Multiply each entry of the result and round to the nearest integer
    *  -# Clamp the resulting int32 values to the [0..255] range and cast to uint8.
    *
    * @param[in]  a               First input tensor  (Matrix A). Data type supported: U8.
    * @param[in]  b               Second input tensor (Matrix B). Data type supported: same as @p a
    * @param[out] output          Output tensor. Data type supported: same as @p a.
    * @param[in]  a_offset        Offset to be added to each element of the matrix A.
    * @param[in]  b_offset        Offset to be added to each element of the matrix B.
    * @param[in]  output_offset   Offset to be added to each element of the output matrix
    * @param[in]  output_mult_int Value to be multiplied to each element of the output matrix
    * @param[in]  shift           Number of bits to shift right the result.
    */
    void configure(const ITensor *a, const ITensor *b, ITensor *output, int32_t a_offset, int32_t b_offset, int32_t output_offset, int32_t output_mult_int, int32_t shift);
    // Inherited methods overridden:
    void run() override;

private:
    NEGEMMInterleave4x4Kernel      _interleave_kernel;
    NEGEMMTranspose1xWKernel       _transpose_kernel;
    NEGEMMLowpMatrixMultiplyKernel _mm_kernel;
    Tensor                         _tmp_a;
    Tensor                         _tmp_b;
};
}
#endif /*__ARM_COMPUTE_NEGEMMLOWP_H__ */
