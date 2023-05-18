/*
 * Copyright (c) 2023 Arm Limited.
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

#ifndef ACL_SRC_CORE_UTILS_QUANTIZATION_ASYMMHELPERS_H
#define ACL_SRC_CORE_UTILS_QUANTIZATION_ASYMMHELPERS_H

#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace quantization {

/** Get minimum and maximum output of the activation function after quantization.
 *
 * Only ReLU, upper bounded ReLU and lower+upper bounded ReLU are supported.
 *
 * @param[in] q_info    Output quantization info.
 * @param[in] act_info  Activation function information.
 * @param[in] data_type Output data type (either QASYMM8 or QASYMM8_SIGNED).
 *
 * @return The minimum and maximum output of the activation function after quantization.
 */
std::tuple<int32_t, int32_t> get_quantized_asymmetric_output_min_max(const QuantizationInfo &q_info, const ActivationLayerInfo &act_info, DataType data_type);

} // namespace quantization
} // namespace arm_compute

#endif // ACL_SRC_CORE_UTILS_QUANTIZATION_ASYMMHELPERS_H
