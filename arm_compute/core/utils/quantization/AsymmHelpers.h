/*
 * Copyright (c) 2017-2021, 2023 Arm Limited.
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
#ifndef ARM_COMPUTE_QUANTIZATION_ASYMM_HELPERS_H
#define ARM_COMPUTE_QUANTIZATION_ASYMM_HELPERS_H

#include "arm_compute/core/Error.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace quantization
{
/** Calculate quantized representation of multiplier.
 *
 * @param[in]  multiplier       Real multiplier.
 * @param[out] quant_multiplier Integer multiplier.
 * @param[out] shift            bit shift. A negative value indicates a left shift, while a positive value indicates a right shift
 * @param[in]  ignore_epsilon   When true, ignore pre-defined epsilon value. Defaults to false
 *
 * @return a status
 */
Status calculate_quantized_multiplier(float multiplier, int32_t *quant_multiplier, int32_t *shift, bool ignore_epsilon = false);
/** Calculate quantized representation of multiplier with value less than one.
 *
 * @param[in]  multiplier       Real multiplier.
 * @param[out] quant_multiplier Integer multiplier.
 * @param[out] right_shift      Right bit shift.
 * @param[in]  ignore_epsilon   When true, ignore pre-defined epsilon value. Defaults to false
 *
 * @return a status
 */
Status calculate_quantized_multiplier_less_than_one(float multiplier, int32_t *quant_multiplier, int32_t *right_shift, bool ignore_epsilon = false);
/** Calculate quantized representation of multiplier having value greater than one.
 *
 * @param[in]  multiplier           Real multiplier.
 * @param[out] quantized_multiplier Integer multiplier.
 * @param[out] left_shift           Left bit shift.
 *
 * @return a status
 */
Status calculate_quantized_multiplier_greater_than_one(float multiplier, int32_t *quantized_multiplier, int32_t *left_shift);

/** Calculate quantized representation of per-channel multipliers
 *
 * @param[in]      iq_info    Input quantization info.
 * @param[in]      wq_info    Weights quantization info.
 * @param[in]      oq_info    Output quantization info.
 * @param[in, out] stage_info GemmLowp output stage info
 *
 * @return a status
 */
Status calculate_quantized_multipliers(const QuantizationInfo &iq_info,
                                       const QuantizationInfo &wq_info,
                                       const QuantizationInfo &oq_info,
                                       GEMMLowpOutputStageInfo &stage_info);

/** Get minimum and maximum values for the input quantized data type
 *
 * @return min and max values for the quantized data type
 */
std::pair<int, int> get_min_max_values_from_quantized_data_type(DataType data_type);

/** Compute quantized per-channel multipliers and shifts. As many multipliers
 * and shifts as output channels are computed. If weights are not quantized
 * per-channel, multipliers and shifts will end up being the same for each
 * channel.
 *
 * @param[in]  input                  Input tensor info.
 * @param[in]  weights                Weights tensor info.
 * @param[in]  output                 Output tensor info.
 * @param[out] output_multipliers_ptr Pointer to the buffer where to store per-channel multipliers.
 * @param[out] output_shifts_ptr      Pointer to the buffer where to store per-channel shifts.
 */
void compute_quantized_multipliers_and_shifts(const ITensorInfo *input,
                                              const ITensorInfo *weights,
                                              const ITensorInfo *output,
                                              int32_t           *output_multipliers_ptr,
                                              int32_t           *output_shifts_ptr);

/** Round to the nearest division by a power-of-two using exponent, copied from NEMath
 *
 * @note This function calculates the following expression: (x + 2^n -1 ) / 2^n where n = exponent
 *
 * @param[in] x        Element to divide.
 * @param[in] exponent Integer value used to round to nearest division by a power-of-two
 *
 * @return the nearest division by a power-of-two using exponent
 */
int32_t rounding_divide_by_pow2(int32_t x, int exponent);

/** Compute multiplication of two integers
 *
 * @param[in] a One integer to multiply
 * @param[in] b Another integer to multiply
 *
 * @return The multiplied value
 */
int32_t saturating_rounding_doubling_highmul(int32_t a, int32_t b);

/** Compute the value multiplied by given quantized multiplier and shift
 *
 * @param[in] input Target value to multiply.
 * @param[in] qmul  Quantized multipler
 * @param[in] shift Left bit shift
 *
 * @return The multiplied value
 */
int32_t multiply_by_quantized_multiplier(int32_t input, int32_t qmul, int32_t shift);

/** Compute the value multiplied the power-of-two
 *
 * @param[in] exponent Exponent used to calculate power-of-two
 * @param[in] v        Target value to multiply
 *
 * @return The multiplied value
 */
int32_t saturating_rounding_multiply_by_pow2(int32_t exponent, int32_t v);

/** Compute quantized multiplier and shift for the inverse square root of input.
 *  Using 3-bit fixed point and 5 iteration of Newton-Raphson method.
 *
 * @param[in]  input           Input to use
 * @param[in]  reverse_shift   -1 to reverse the shift direction
 * @param[out] output_inv_sqrt Quantized multiplier for inverse square root
 * @param[out] output_shift    Shift for inverse square root
 *
 */
void get_invsqrt_quantized_multiplier_exp(int32_t input, int32_t reverse_shift, int32_t &output_inv_sqrt, int32_t &output_shift);

} // namespace quantization
} // namespace arm_compute
#endif /* ARM_COMPUTE_IO_FILE_HANDLER_H */
