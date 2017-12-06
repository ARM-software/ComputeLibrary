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
#ifndef __ARM_COMPUTE_CLGEMMLOWPOUTPUTSTAGE_H__
#define __ARM_COMPUTE_CLGEMMLOWPOUTPUTSTAGE_H__

#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

/** This file contains all available output stages for GEMMLowp on OpenCL.
 *
 *  In gemmlowp, the "output stage" is the process that takes a final int32 accumulator value (the output of @ref CLGEMMLowpMatrixMultiplyCore),
 *  and processes it to obtain the final ASYMM8 value.
 *
 *  More information about the GEMMLowp output stage can be found at https://github.com/google/gemmlowp/blob/master/doc/output.md
 */

namespace arm_compute
{
class ITensor;

/** Basic function to execute CLGEMMLowpQuantizeDownInt32ToUint8Scale on OpenCL.
 *
 *  CLGEMMLowpQuantizeDownInt32ToUint8Scale depends on 3 parameters: result_offset, result_mult_int, result_shift
 *  The final result is:
 *
 *  ((input[i][k] + result_offset) * result_mult_int) >> result_shift
 *
 * In case the bias tensor is provided, the final result is:
 *
 *  ((input[i][k] + bias[k] + result_offset) * result_mult_int) >> result_shift
 *
 *  This function calls the following OpenCL kernels:
 *
 * -# @ref CLGEMMLowpQuantizeDownInt32ToUint8ScaleKernel
 *
 * @note The function accepts also 2 optional input arguments (min and max) which can be used to implement "rectified linear unit" activation functions
 *       after the result is shifted right by result_shift
*/
class CLGEMMLowpQuantizeDownInt32ToUint8Scale : public ICLSimpleFunction
{
public:
    /** Initialise the kernel's inputs, output
     *
     * @param[in]  input           Input tensor. It is the output of @ref CLGEMMLowpMatrixMultiplyCore function. Data type supported: S32
     * @param[in]  bias            Biases tensor. Only shared biases supported and it can be a nullptr if the addition of biases is not required.
     *                             Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[out] output          Output tensor. Data type supported: Data type supported: QASYMM8
     * @param[in]  result_offset   Offset to be added to each element of the input matrix
     * @param[in]  result_mult_int Value to be multiplied to each element of the input matrix when once the result_offset has been add
     * @param[in]  result_shift    Number of bits to shift right the result before converting back to QASYMM8
     * @param[in]  min             (Optional) Min value used to saturate down the output result before converting back to QASYMM8
     * @param[in]  max             (Optional) Max value used to saturate up the output result before converting back to QASYMM8,
     *                             Along with @p min, this value can be used to implement "rectified linear unit" activation functions
     */
    void configure(const ICLTensor *input, const ICLTensor *bias, ICLTensor *output, int result_offset, int result_mult_int, int result_shift, int min = 0, int max = 0);
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMLowpQuantizeDownInt32ToUint8Scale
     *
     * @param[in] input  Input tensor. It is the output of @ref CLGEMMLowpMatrixMultiplyCore function. Data type supported: S32
     * @param[in] bias   Biases tensor. Only shared biases supported and it can be a nullptr if the addition of biases is not required.
     *                   Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[in] output Output tensor. Data type supported: Data type supported: QASYMM8
     * @param[in] min    (Optional) Min value used to saturate down the output result before converting back to QASYMM8
     * @param[in] max    (Optional) Max value used to saturate up the output result before converting back to QASYMM8,
     *                   Along with @p min, this value can be used to implement "rectified linear unit" activation functions
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, int min = 0, int max = 0);
};

/** Basic function to execute CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint on OpenCL.
 *
 *  CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint depends on 3 parameters:
 *
 *  result_fixedpoint_multiplier, result_shift, result_offset_after_shift
 *
 * The final result is:
 *
 * (FixedPointMul(input[i][k], result_fixedpoint_multiplier) >> result_shift) + result_offset_after_shift
 *
 * where FixedPointMul(x, y) is the nearest integer to the following
 * mathematical expression, evaluated without overflow or intermediate rounding:
 *
 * (x * y) / 2^31
 *
 * For more information: https://github.com/google/gemmlowp/blob/master/public/output_stages.h#L68
 *
 * In case the bias tensor is provided, the final result is:
 *
 * ((FixedPointMul(input[i][k] + bias[k], result_fixedpoint_multiplier)) >> result_shift) + result_offset_after_shift
 *
 *  This function calls the following OpenCL kernels:
 *
 * -# @ref CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel
 *
 * @note The function accepts also 2 optional input arguments (min and max) which can be used to implement "rectified linear unit" activation functions
 *       after the result is shifted right by result_shift
*/
class CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint : public ICLSimpleFunction
{
public:
    /** Initialise the kernel's inputs, output
     *
     * @param[in]  input                        Input tensor. Data type supported: S32
     * @param[in]  bias                         Biases tensor. Only shared biases supported and it can be a nullptr if the biases addition is not required.
     *                                          Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[out] output                       Output tensor. Data type supported: Data type supported: QASYMM8
     * @param[in]  result_fixedpoint_multiplier Fixed point value to be multiplied to each element of the input matrix when once the result_offset has been add
     * @param[in]  result_shift                 Number of bits to shift right the result after the fixed point multiplication
     * @param[in]  result_offset_after_shift    Offset to be applied to result before converting it back to QASYMM8
     * @param[in]  min                          (Optional) Min value used to saturate down the output result before converting back to QASYMM8
     * @param[in]  max                          (Optional) Max value used to saturate up the output result before converting back to QASYMM8,
     *                                          Along with @p min, this value can be used to implement "rectified linear unit" activation functions
     */
    void configure(const ICLTensor *input, const ICLTensor *bias, ICLTensor *output, int result_fixedpoint_multiplier, int result_shift, int result_offset_after_shift, int min = 0, int max = 0);
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint
     *
     * @param[in] input  Input tensor. It is the output of @ref CLGEMMLowpMatrixMultiplyCore function. Data type supported: S32
     * @param[in] bias   Biases tensor. Only shared biases supported and it can be a nullptr if the addition of biases is not required.
     *                   Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[in] output Output tensor. Data type supported: Data type supported: QASYMM8
     * @param[in] min    (Optional) Min value used to saturate down the output result before converting back to QASYMM8
     * @param[in] max    (Optional) Max value used to saturate up the output result before converting back to QASYMM8,
     *                   Along with @p min, this value can be used to implement "rectified linear unit" activation functions
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, int min = 0, int max = 0);
};
}
#endif /*__ARM_COMPUTE_CLGEMMLOWPOUTPUTSTAGE_H__ */