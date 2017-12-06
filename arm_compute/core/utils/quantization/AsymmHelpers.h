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
#ifndef __ARM_COMPUTE_QUANTIZATION_ASYMM_HELPERS_H__
#define __ARM_COMPUTE_QUANTIZATION_ASYMM_HELPERS_H__

#include "arm_compute/core/Error.h"

namespace arm_compute
{
namespace quantization
{
/** Calculate quantized representation of multiplier with value less than one.
 *
 * @param[in]  multiplier       Real multiplier.
 * @param[out] quant_multiplier Integer multiplier.
 * @param[out] right_shift      Right bit shift.
 *
 * @return a status
 */
arm_compute::Status calculate_quantized_multiplier_less_than_one(double multiplier, int *quant_multiplier, int *right_shift);
/** Calculate quantized representation of multiplier having value greater than one.
 *
 * @param[in]  multiplier           Real multiplier.
 * @param[out] quantized_multiplier Integer multiplier.
 * @param[out] left_shift           Left bit shift.
 *
 * @return a status
 */
arm_compute::Status calculate_quantized_multiplier_greater_than_one(double multiplier, int *quantized_multiplier, int *left_shift);
} // namespace quantization
} // namespace arm_compute
#endif /* __ARM_COMPUTE_IO_FILE_HANDLER_H__ */
