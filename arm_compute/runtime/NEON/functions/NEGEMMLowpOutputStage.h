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
#ifndef ARM_COMPUTE_NEGEMMLOWPOUTPUTSTAGE_H
#define ARM_COMPUTE_NEGEMMLOWPOUTPUTSTAGE_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/INESimpleFunctionNoBorder.h"

/** This file contains all available output stages for GEMMLowp on Neon.
 *
 *  In gemmlowp, the "output stage" is the process that takes a final int32 accumulator value (the output of @ref NEGEMMLowpMatrixMultiplyCore),
 *  and processes it to obtain the final ASYMM8 value.
 *
 *  More information about the GEMMLowp output stage can be found at https://github.com/google/gemmlowp/blob/master/doc/output.md
 */

namespace arm_compute
{
class ITensor;
class ITensorInfo;

/** Basic function to execute NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint on Neon.
 *
 *  NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint depends on 3 parameters:
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
 *  This function calls the following Neon kernels:
 *
 * -# @ref NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel
 *
 * @note The function accepts also 2 optional input arguments (min and max) which can be used to implement "rectified linear unit" activation functions
 *       after the result is shifted right by result_shift
*/
class NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint : public INESimpleFunctionNoBorder
{
public:
    /** Constructor */
    NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint(const NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint &operator=(const NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint(NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint &operator=(NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint &&) = delete;
    /** Default destructor */
    ~NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint();
    /** Initialise the kernel's inputs, output
     *
     * @param[in]  input                        Input tensor. Data type supported: S32
     * @param[in]  bias                         Biases tensor. Only shared biases supported and it can be a nullptr if the biases addition is not required.
     *                                          Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[out] output                       Output tensor. Data type supported: Data type supported: QASYMM8
     * @param[in]  result_fixedpoint_multiplier Fixed point value to be multiplied to each element of the input matrix when once the result_offset has been add
     * @param[in]  result_shift                 Number of bits to shift right the result after the fixed point multiplication
     * @param[in]  result_offset_after_shift    Offset to be applied to result before converting it back to QASYMM8
     * @param[in]  min                          (Optional) Min value used to saturate down the output result before converting back to QASYMM8. Defaults to the minimum possible 32-bit signed integer.
     * @param[in]  max                          (Optional) Max value used to saturate up the output result before converting back to QASYMM8,
     *                                          Along with @p min, this value can be used to implement "rectified linear unit" activation functions. Defaults to the maximum possible 32-bit signed integer.
     */
    void configure(const ITensor *input, const ITensor *bias, ITensor *output, int result_fixedpoint_multiplier, int result_shift, int result_offset_after_shift,
                   int min = std::numeric_limits<int32_t>::lowest(), int max = std::numeric_limits<int32_t>::max());
    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint
     *
     * @param[in] input  Input tensor. It is the output of @ref NEGEMMLowpMatrixMultiplyCore function. Data type supported: S32
     * @param[in] bias   Biases tensor. Only shared biases supported and it can be a nullptr if the addition of biases is not required.
     *                   Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[in] output Output tensor. Data type supported: Data type supported: QASYMM8
     * @param[in] min    (Optional) Min value used to saturate down the output result before converting back to QASYMM8. Defaults to the minimum possible 32-bit signed integer.
     * @param[in] max    (Optional) Max value used to saturate up the output result before converting back to QASYMM8,
     *                            Along with @p min, this value can be used to implement "rectified linear unit" activation functions. Defaults to the maximum possible 32-bit signed integer.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, int min = std::numeric_limits<int32_t>::lowest(), int max = std::numeric_limits<int32_t>::max());
};
/** Basic function to execute NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint on Neon.
 *
 *  NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint depends on 3 parameters:
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
 *  This function calls the following Neon kernels:
 *
 * -# @ref NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel
 *
 * @note The function accepts also 2 optional input arguments (min and max) which can be used to implement "rectified linear unit" activation functions
 *       after the result is shifted right by result_shift
*/
class NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint : public INESimpleFunctionNoBorder
{
public:
    /** Constructor */
    NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint(const NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint &operator=(const NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint(NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint &operator=(NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint &&) = delete;
    /** Default destructor */
    ~NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint();
    /** Initialise the kernel's inputs, output
     *
     * @param[in]  input                        Input tensor. Data type supported: S32
     * @param[in]  bias                         Biases tensor. Only shared biases supported and it can be a nullptr if the biases addition is not required.
     *                                          Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[out] output                       Output tensor. Data type supported: Data type supported: QASYMM8_SIGNED
     * @param[in]  result_fixedpoint_multiplier Fixed point value to be multiplied to each element of the input matrix when once the result_offset has been add
     * @param[in]  result_shift                 Number of bits to shift right the result after the fixed point multiplication
     * @param[in]  result_offset_after_shift    Offset to be applied to result before converting it back to QASYMM8_SIGNED
     * @param[in]  min                          (Optional) Min value used to saturate down the output result before converting back to QASYMM8_SIGNED. Defaults to the minimum possible 32-bit signed integer.
     * @param[in]  max                          (Optional) Max value used to saturate up the output result before converting back to QASYMM8_SIGNED,
     *                                          Along with @p min, this value can be used to implement "rectified linear unit" activation functions. Defaults to the maximum possible 32-bit signed integer.
     */
    void configure(const ITensor *input, const ITensor *bias, ITensor *output, int result_fixedpoint_multiplier, int result_shift, int result_offset_after_shift,
                   int min = std::numeric_limits<int32_t>::lowest(), int max = std::numeric_limits<int32_t>::max());
    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPoint
     *
     * @param[in] input  Input tensor. It is the output of @ref NEGEMMLowpMatrixMultiplyCore function. Data type supported: S32
     * @param[in] bias   Biases tensor. Only shared biases supported and it can be a nullptr if the addition of biases is not required.
     *                   Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[in] output Output tensor. Data type supported: Data type supported: QASYMM8_SIGNED
     * @param[in] min    (Optional) Min value used to saturate down the output result before converting back to QASYMM8_SIGNED. Defaults to the minimum possible 32-bit signed integer.
     * @param[in] max    (Optional) Max value used to saturate up the output result before converting back to QASYMM8_SIGNED,
     *                            Along with @p min, this value can be used to implement "rectified linear unit" activation functions. Defaults to the maximum possible 32-bit signed integer.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, int min = std::numeric_limits<int32_t>::lowest(), int max = std::numeric_limits<int32_t>::max());
};
/** Basic function to execute NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint on Neon.
 *
 *  NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint depends on 2 parameters:
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
 *  This function calls the following Neon kernels:
 *
 * -# @ref NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel
 *
 * @note The function accepts also 2 optional input arguments (min and max) which can be used to implement "rectified linear unit" activation functions
 *       after the result is shifted right by result_shift
*/
class NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint : public INESimpleFunctionNoBorder
{
public:
    /** Constructor */
    NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint(const NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint &operator=(const NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint(NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint &operator=(NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint &&) = delete;
    /** Default destructor */
    ~NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint();
    /** Initialise the kernel's inputs, output
     *
     * @param[in]  input                        Input tensor. Data type supported: S32
     * @param[in]  bias                         Biases tensor. Only shared biases supported and it can be a nullptr if the biases addition is not required.
     *                                          Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[out] output                       Output tensor. Data type supported: Data type supported: QSYMM16
     * @param[in]  result_fixedpoint_multiplier Fixed point value to be multiplied to each element of the input matrix when once the result_offset has been add
     * @param[in]  result_shift                 Number of bits to shift right the result after the fixed point multiplication
     * @param[in]  min                          (Optional) Min value used to saturate down the output result before converting back to QSYMM16. Defaults to the minimum possible 32-bit signed integer.
     * @param[in]  max                          (Optional) Max value used to saturate up the output result before converting back to QSYMM16.
     *                                          Along with @p min, this value can be used to implement "rectified linear unit" activation functions. Defaults to the maximum possible 32-bit signed integer.
     */
    void configure(const ITensor *input, const ITensor *bias, ITensor *output, int result_fixedpoint_multiplier, int result_shift, int min = std::numeric_limits<int32_t>::lowest(),
                   int max = std::numeric_limits<int32_t>::max());
    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint
     *
     * @param[in] input  Input tensor info. It is the output of @ref NEGEMMLowpMatrixMultiplyCore function. Data type supported: S32
     * @param[in] bias   Biases tensor info. Only shared biases supported and it can be a nullptr if the addition of biases is not required.
     *                   Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[in] output Output tensor info. Data type supported: Data type supported: QSYMM16
     * @param[in] min    (Optional) Min value used to saturate down the output result before converting back to QSYMM16. Defaults to the minimum possible 32-bit signed integer.
     * @param[in] max    (Optional) Max value used to saturate up the output result before converting back to QSYMM16,
     *                            Along with @p min, this value can be used to implement "rectified linear unit" activation functions. Defaults to the maximum possible 32-bit signed integer.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, int min = std::numeric_limits<int32_t>::lowest(), int max = std::numeric_limits<int32_t>::max());
};

/** Basic function to execute GEMMLowpQuantizeDown kernels on Neon.
 *
 *  This function calls the following Neon kernels:
 *
 * -# @ref NEGEMMLowpQuantizeDownInt32ScaleKernel
 * -# @ref NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel
 * -# @ref NEGEMMLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel
 * -# @ref NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel
*/
class NEGEMMLowpOutputStage : public INESimpleFunctionNoBorder
{
public:
    /** Constructor */
    NEGEMMLowpOutputStage() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMLowpOutputStage(const NEGEMMLowpOutputStage &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMLowpOutputStage &operator=(const NEGEMMLowpOutputStage &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEGEMMLowpOutputStage(NEGEMMLowpOutputStage &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEGEMMLowpOutputStage &operator=(NEGEMMLowpOutputStage &&) = delete;
    /** Default destructor */
    ~NEGEMMLowpOutputStage();
    /** Initialise the kernel's inputs, output
     *
     * @param[in]  input  Input tensor. Data type supported: S32
     * @param[in]  bias   Biases tensor. Only shared biases supported and it can be a nullptr if the biases addition is not required.
     *                    Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[out] output Output tensor. Data type supported: Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM16
     * @param[in]  info   GEMMLowp output stage metadata.
     */
    void configure(const ITensor *input, const ITensor *bias, ITensor *output, const GEMMLowpOutputStageInfo &info);
    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMLowpOutputStage
     *
     * @param[in] input  Input tensor info. It is the output of @ref NEGEMMLowpMatrixMultiplyCore function. Data type supported: S32
     * @param[in] bias   Biases tensor info. Only shared biases supported and it can be a nullptr if the addition of biases is not required.
     *                   Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[in] output Output tensor info. Data type supported: Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM16
     * @param[in] info   GEMMLowp output stage metadata.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, const GEMMLowpOutputStageInfo &info);
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEGEMMLOWPOUTPUTSTAGE_H */
