/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLGEMMLOWPOUTPUTSTAGE_H
#define ARM_COMPUTE_CLGEMMLOWPOUTPUTSTAGE_H

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

#include <limits>

/** This file contains all available output stages for GEMMLowp on OpenCL.
 *
 *  In gemmlowp, the "output stage" is the process that takes a final int32 accumulator value (the output of @ref CLGEMMLowpMatrixMultiplyCore),
 *  and processes it to obtain the final QASYMM8/QASYMM8_SIGNED value.
 *
 *  More information about the GEMMLowp output stage can be found at https://github.com/google/gemmlowp/blob/master/doc/output.md
 */

namespace arm_compute
{
class CLCompileContext;
class ITensor;
class ICLTensor;
class ITensorInfo;
struct GEMMLowpOutputStageInfo;

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
 * -# @ref CLGEMMLowpQuantizeDownInt32ScaleByFixedPointKernel
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
     * @param[out] output                       Output tensor. Data type supported: QASYMM8
     * @param[in]  result_fixedpoint_multiplier Fixed point value to be multiplied to each element of the input matrix when once the result_offset has been add
     * @param[in]  result_shift                 Number of bits to shift right the result after the fixed point multiplication
     * @param[in]  result_offset_after_shift    Offset to be applied to result before converting it back to QASYMM8
     * @param[in]  min                          (Optional) Min value used to saturate down the output result before converting back to QASYMM8. Defaults to the minimum possible 32-bit signed integer.
     * @param[in]  max                          (Optional) Max value used to saturate up the output result before converting back to QASYMM8,
     *                                          Along with @p min, this value can be used to implement "rectified linear unit" activation functions. Defaults to the maximum possible 32-bit signed integer.
     */
    void configure(const ICLTensor *input, const ICLTensor *bias, ICLTensor *output, int result_fixedpoint_multiplier, int result_shift, int result_offset_after_shift,
                   int min = std::numeric_limits<int32_t>::lowest(), int max = std::numeric_limits<int32_t>::max());
    /** Initialise the kernel's inputs, output
     *
     * @param[in]  compile_context              The compile context to be used.
     * @param[in]  input                        Input tensor. Data type supported: S32
     * @param[in]  bias                         Biases tensor. Only shared biases supported and it can be a nullptr if the biases addition is not required.
     *                                          Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[out] output                       Output tensor. Data type supported: QASYMM8
     * @param[in]  result_fixedpoint_multiplier Fixed point value to be multiplied to each element of the input matrix when once the result_offset has been add
     * @param[in]  result_shift                 Number of bits to shift right the result after the fixed point multiplication
     * @param[in]  result_offset_after_shift    Offset to be applied to result before converting it back to QASYMM8
     * @param[in]  min                          (Optional) Min value used to saturate down the output result before converting back to QASYMM8. Defaults to the minimum possible 32-bit signed integer.
     * @param[in]  max                          (Optional) Max value used to saturate up the output result before converting back to QASYMM8,
     *                                          Along with @p min, this value can be used to implement "rectified linear unit" activation functions. Defaults to the maximum possible 32-bit signed integer.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *bias, ICLTensor *output, int result_fixedpoint_multiplier, int result_shift,
                   int result_offset_after_shift,
                   int min = std::numeric_limits<int32_t>::lowest(), int max = std::numeric_limits<int32_t>::max());
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint
     *
     * @param[in] input  Input tensor. It is the output of @ref CLGEMMLowpMatrixMultiplyCore function. Data type supported: S32
     * @param[in] bias   Biases tensor. Only shared biases supported and it can be a nullptr if the addition of biases is not required.
     *                   Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[in] output Output tensor. Data type supported: QASYMM8
     * @param[in] min    (Optional) Min value used to saturate down the output result before converting back to QASYMM8. Defaults to the minimum possible 32-bit signed integer.
     * @param[in] max    (Optional) Max value used to saturate up the output result before converting back to QASYMM8,
     *                            Along with @p min, this value can be used to implement "rectified linear unit" activation functions. Defaults to the maximum possible 32-bit signed integer.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, int min = std::numeric_limits<int32_t>::lowest(), int max = std::numeric_limits<int32_t>::max());
};

/** Basic function to execute CLGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint on OpenCL.
 *
 *  CLGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint depends on 3 parameters:
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
 * -# @ref CLGEMMLowpQuantizeDownInt32ScaleByFixedPointKernel
 *
 * @note The function accepts also 2 optional input arguments (min and max) which can be used to implement "rectified linear unit" activation functions
 *       after the result is shifted right by result_shift
*/
class CLGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint : public ICLSimpleFunction
{
public:
    /** Initialise the kernel's inputs, output
     *
     * @param[in]  input                        Input tensor. Data type supported: S32
     * @param[in]  bias                         Biases tensor. Only shared biases supported and it can be a nullptr if the biases addition is not required.
     *                                          Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[out] output                       Output tensor. Data type supported: QASYMM8_SIGNED
     * @param[in]  result_fixedpoint_multiplier Fixed point value to be multiplied to each element of the input matrix when once the result_offset has been add
     * @param[in]  result_shift                 Number of bits to shift right the result after the fixed point multiplication
     * @param[in]  result_offset_after_shift    Offset to be applied to result before converting it back to QASYMM8_SIGNED
     * @param[in]  min                          (Optional) Min value used to saturate down the output result before converting back to QASYMM8_SIGNED. Defaults to the minimum possible 32-bit signed integer.
     * @param[in]  max                          (Optional) Max value used to saturate up the output result before converting back to QASYMM8_SIGNED. Defaults to 0
     *                                          Along with @p min, this value can be used to implement "rectified linear unit" activation functions. Defaults to the maximum possible 32-bit signed integer.
     */
    void configure(const ICLTensor *input, const ICLTensor *bias, ICLTensor *output, int result_fixedpoint_multiplier, int result_shift, int result_offset_after_shift,
                   int min = std::numeric_limits<int32_t>::lowest(), int max = std::numeric_limits<int32_t>::max());
    /** Initialise the kernel's inputs, output
     *
     * @param[in]  compile_context              The compile context to be used.
     * @param[in]  input                        Input tensor. Data type supported: S32
     * @param[in]  bias                         Biases tensor. Only shared biases supported and it can be a nullptr if the biases addition is not required.
     *                                          Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[out] output                       Output tensor. Data type supported: QASYMM8_SIGNED
     * @param[in]  result_fixedpoint_multiplier Fixed point value to be multiplied to each element of the input matrix when once the result_offset has been add
     * @param[in]  result_shift                 Number of bits to shift right the result after the fixed point multiplication
     * @param[in]  result_offset_after_shift    Offset to be applied to result before converting it back to QASYMM8_SIGNED
     * @param[in]  min                          (Optional) Min value used to saturate down the output result before converting back to QASYMM8_SIGNED. Defaults to the minimum possible 32-bit signed integer.
     * @param[in]  max                          (Optional) Max value used to saturate up the output result before converting back to QASYMM8_SIGNED. Defaults to 0
     *                                          Along with @p min, this value can be used to implement "rectified linear unit" activation functions. Defaults to the maximum possible 32-bit signed integer.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *bias, ICLTensor *output, int result_fixedpoint_multiplier, int result_shift,
                   int result_offset_after_shift,
                   int min = std::numeric_limits<int32_t>::lowest(), int max = std::numeric_limits<int32_t>::max());
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint
     *
     * @param[in] input  Input tensor. It is the output of @ref CLGEMMLowpMatrixMultiplyCore function. Data type supported: S32
     * @param[in] bias   Biases tensor. Only shared biases supported and it can be a nullptr if the addition of biases is not required.
     *                   Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[in] output Output tensor. Data type supported: QASYMM8_SIGNED
     * @param[in] min    (Optional) Min value used to saturate down the output result before converting back to QASYMM8_SIGNED. Defaults to the minimum possible 32-bit signed integer.
     * @param[in] max    (Optional) Max value used to saturate up the output result before converting back to QASYMM8_SIGNED. Defaults to 0
     *                            Along with @p min, this value can be used to implement "rectified linear unit" activation functions. Defaults to the maximum possible 32-bit signed integer.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, int min = std::numeric_limits<int32_t>::lowest(), int max = std::numeric_limits<int32_t>::max());
};

/** Basic function to execute CLGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint on OpenCL.
 *
 *  CLGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint depends on 2 parameters:
 *
 *  result_fixedpoint_multiplier, result_shift
 *
 * The final result is:
 *
 * (FixedPointMul(input[i][k], result_fixedpoint_multiplier) >> result_shift)
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
 *  This function calls the following CL kernels:
 *
 * -# @ref CLGEMMLowpQuantizeDownInt32ScaleByFixedPointKernel
 *
 * @note The function accepts also 2 optional input arguments (min and max) which can be used to implement "rectified linear unit" activation functions
 *       after the result is shifted right by result_shift
*/
class CLGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint : public ICLSimpleFunction
{
public:
    /** Initialise the kernel's inputs, output
     *
     * @param[in]  input                        Input tensor. Data type supported: S32
     * @param[in]  bias                         Biases tensor. Only shared biases supported and it can be a nullptr if the biases addition is not required.
     *                                          Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[out] output                       Output tensor. Data type supported: QSYMM16
     * @param[in]  result_fixedpoint_multiplier Fixed point value to be multiplied to each element of the input matrix when once the result_offset has been add
     * @param[in]  result_shift                 Number of bits to shift right the result after the fixed point multiplication
     * @param[in]  min                          (Optional) Min value used to saturate down the output result before converting back to QSYMM16. Defaults to the minimum possible 32-bit signed integer.
     * @param[in]  max                          (Optional) Max value used to saturate up the output result before converting back to QSYMM16.
     *                                          Along with @p min, this value can be used to implement "rectified linear unit" activation functions. Defaults to the maximum possible 32-bit signed integer.
     */
    void configure(const ICLTensor *input, const ICLTensor *bias, ICLTensor *output, int result_fixedpoint_multiplier, int result_shift, int min = std::numeric_limits<int32_t>::lowest(),
                   int max = std::numeric_limits<int32_t>::max());
    /** Initialise the kernel's inputs, output
     *
     * @param[in]  compile_context              The compile context to be used.
     * @param[in]  input                        Input tensor. Data type supported: S32
     * @param[in]  bias                         Biases tensor. Only shared biases supported and it can be a nullptr if the biases addition is not required.
     *                                          Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[out] output                       Output tensor. Data type supported: QSYMM16
     * @param[in]  result_fixedpoint_multiplier Fixed point value to be multiplied to each element of the input matrix when once the result_offset has been add
     * @param[in]  result_shift                 Number of bits to shift right the result after the fixed point multiplication
     * @param[in]  min                          (Optional) Min value used to saturate down the output result before converting back to QSYMM16. Defaults to the minimum possible 32-bit signed integer.
     * @param[in]  max                          (Optional) Max value used to saturate up the output result before converting back to QSYMM16.
     *                                          Along with @p min, this value can be used to implement "rectified linear unit" activation functions. Defaults to the maximum possible 32-bit signed integer.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *bias, ICLTensor *output, int result_fixedpoint_multiplier, int result_shift,
                   int min = std::numeric_limits<int32_t>::lowest(), int max = std::numeric_limits<int32_t>::max());
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint
     *
     * @param[in] input  Input tensor info. It is the output of @ref CLGEMMLowpMatrixMultiplyCore function. Data type supported: S32
     * @param[in] bias   Biases tensor info. Only shared biases supported and it can be a nullptr if the addition of biases is not required.
     *                   Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[in] output Output tensor info. Data type supported: QSYMM16
     * @param[in] min    (Optional) Min value used to saturate down the output result before converting back to QSYMM16. Defaults to the minimum possible 32-bit signed integer.
     * @param[in] max    (Optional) Max value used to saturate up the output result before converting back to QSYMM16,
     *                            Along with @p min, this value can be used to implement "rectified linear unit" activation functions. Defaults to the maximum possible 32-bit signed integer.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, int min = std::numeric_limits<int32_t>::lowest(), int max = std::numeric_limits<int32_t>::max());
};
/** Basic function to execute GEMMLowpQuantizeDown kernels on CL.
 *
 *  This function calls the following CL kernels:
 *
 * -# @ref CLGEMMLowpQuantizeDownInt32ScaleKernel
 * -# @ref CLGEMMLowpQuantizeDownInt32ScaleByFloatKernel
 * -# @ref CLGEMMLowpQuantizeDownInt32ScaleByFixedPointKernel
*/
class CLGEMMLowpOutputStage : public ICLSimpleFunction
{
public:
    /** Initialise the kernel's inputs, output
     *
     * @param[in]  input  Input tensor. Data type supported: S32
     * @param[in]  bias   Biases tensor. Only shared biases supported and it can be a nullptr if the biases addition is not required.
     *                    Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[out] output Output tensor. Data type supported: QASYMM8/QASYMM8_SIGNED
     * @param[in]  info   GEMMLowp output stage metadata.
     */
    void configure(const ICLTensor *input, const ICLTensor *bias, ICLTensor *output, const GEMMLowpOutputStageInfo &info);
    /** Initialise the kernel's inputs, output
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Input tensor. Data type supported: S32
     * @param[in]  bias            Biases tensor. Only shared biases supported and it can be a nullptr if the biases addition is not required.
     *                             Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[out] output          Output tensor. Data type supported: QASYMM8/QASYMM8_SIGNED
     * @param[in]  info            GEMMLowp output stage metadata.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *bias, ICLTensor *output, const GEMMLowpOutputStageInfo &info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMLowpQuantizeDownInt32ScaleByFixedPointKernel
     *
     * @param[in] input  Input tensor. It is the output of @ref CLGEMMLowpMatrixMultiplyCore function. Data type supported: S32
     * @param[in] bias   Biases tensor. Only shared biases supported and it can be a nullptr if the addition of biases is not required.
     *                   Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[in] output Output tensor. Data type supported: QASYMM8/QASYMM8_SIGNED
     * @param[in] info   GEMMLowp output stage metadata.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, const GEMMLowpOutputStageInfo &info);
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLGEMMLOWPOUTPUTSTAGE_H */
