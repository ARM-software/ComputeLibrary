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
#ifndef __ARM_COMPUTE_NEGEMMLOWPOUTPUTSTAGE_H__
#define __ARM_COMPUTE_NEGEMMLOWPOUTPUTSTAGE_H__

#include "arm_compute/runtime/NEON/INESimpleFunction.h"

/** This file contains all available output stages for GEMMLowp on NEON.
 *
 *  In gemmlowp, the "output stage" is the process that takes a final int32 accumulator value (the output of @ref NEGEMMLowpMatrixMultiplyCore),
 *  and processes it to obtain the final ASYMM8 value.
 *
 *  More information about the GEMMLowp output stage can be found at https://github.com/google/gemmlowp/blob/master/doc/output.md
 */

namespace arm_compute
{
class ITensor;

/** Basic function to execute NEGEMMLowpQuantizeDownInt32ToUint8Scale on NEON.
 *
 *  NEGEMMLowpQuantizeDownInt32ToUint8Scale depends on 3 parameters: result_offset, result_mult_int, result_shift
 *  The final result is:
 *
 *  ((input[i][k] + result_offset) * result_mult_int + rounding) >> result_shift
 *
 *  where rounding = (result_shift < 1) ? 0 : (1 << (result_shift - 1))
 *
 *  This function calls the following NEON kernels:
 *
 * -# @ref NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel
 *
*/
class NEGEMMLowpQuantizeDownInt32ToUint8Scale : public INESimpleFunction
{
public:
    /** Initialise the kernel's inputs, output
    *
    * @param[in]  input           Input tensor. It is the output of @ref NEGEMMLowpMatrixMultiplyCore function. Data type supported: S32
    * @param[out] output          Output tensor. Data type supported: Data type supported: QASYMM8
    * @param[in]  result_offset   Offset to be added to each element of the input matrix
    * @param[in]  result_mult_int Value to be multiplied to each element of the input matrix when once the result_offset has been add
    * @param[in]  result_shift    Number of bits to shift right the result before converting back to QASYMM8
    */
    void configure(const ITensor *input, ITensor *output, int result_offset, int result_mult_int, int result_shift);
};
}
#endif /*__ARM_COMPUTE_NEGEMMLOWPOUTPUTSTAGE_H__ */