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
#ifndef __ARM_COMPUTE_CLGEMMLOWP_H__
#define __ARM_COMPUTE_CLGEMMLOWP_H__

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/CL/kernels/CLGEMMInterleave4x4Kernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMLowpMatrixMultiplyKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMTranspose1xWKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Basic function to execute GEMMLowp on OpenCL. This function calls the following OpenCL kernels:
*
*  -# @ref CLGEMMInterleave4x4Kernel
*  -# @ref CLGEMMTranspose1xWKernel
*  -# @ref CLGEMMLowpMatrixMultiplyKernel
*
*/
class CLGEMMLowp : public IFunction
{
public:
    /** Constructor */
    CLGEMMLowp();
    /** Initialise the kernel's inputs, output
    *
    * @note GEMM_LOWP:  low precision matrix multiply kernel
    *  This kernel performs the following computation:
    *
    *  -# Convert a values from uint8 to int32 and add a_offset to each of them.
    *  -# Convert b values from uint8 to int32 and add b_offset to each of them.
    *  -# Compute the int32 matrix product of the resulting a * b.
    *  -# Add output_offset to each entry of the result.
    *  -# Multiply each entry of the result and round to the nearest integer
    *  -# Clamp the resulting int32 values to the [0..255] range and cast to uint8.
    *
    * @param[in]  a               First input tensor  (Matrix A). Data types supported: U8.
    * @param[in]  b               Second input tensor (Matrix B). Data types supported: same as @p a.
    * @param[out] output          Output tensor. Data types supported: same as @p a.
    * @param[in]  a_offset        Offset to be added to each element of the matrix A.
    * @param[in]  b_offset        Offset to be added to each element of the matrix B.
    * @param[in]  output_offset   Offset to be added to each element of the output matrix
    * @param[in]  output_mult_int Multiplied with each element of the output matrix
    * @param[in]  shift           Number of bits to shift right the result.
    */
    void configure(const ICLTensor *a, const ICLTensor *b, ICLTensor *output, int32_t a_offset, int32_t b_offset, int32_t output_offset, int32_t output_mult_int, int32_t shift);

    // Inherited methods overridden:
    void run() override;

private:
    CLGEMMInterleave4x4Kernel      _interleave_kernel;
    CLGEMMTranspose1xWKernel       _transpose_kernel;
    CLGEMMLowpMatrixMultiplyKernel _mm_kernel;
    CLTensor                       _tmp_a;
    CLTensor                       _tmp_b;
};
}
#endif /*__ARM_COMPUTE_CLGEMMLOWP_H__ */
