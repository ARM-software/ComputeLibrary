/*
 * Copyright (c) 2017-2025 Arm Limited.
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

#ifndef ACL_TESTS_VALIDATION_HELPERS_ACTIVATIONHELPERS_H
#define ACL_TESTS_VALIDATION_HELPERS_ACTIVATIONHELPERS_H

#include "arm_compute/core/CoreTypes.h"
#include "arm_compute/core/QuantizationInfo.h"
#include "arm_compute/function_info/ActivationLayerInfo.h"

#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace helper
{
/** Define relative tolerance of the activation layer.
 *
 * @param[in] data_type  The data type used.
 * @param[in] activation The activation function used.
 *
 * @return Relative tolerance depending on the activation function.
 */
RelativeTolerance<float> relative_tolerance(DataType data_type, ActivationLayerInfo::ActivationFunction activation);

/** Define absolute tolerance of the activation layer.
 *
 * Similar to @ref arm_compute::test::validation::relative_tolerance()
 *
 * @return Absolute tolerance depending on the activation function.
 */
AbsoluteTolerance<float> absolute_tolerance(DataType data_type, ActivationLayerInfo::ActivationFunction activation);

/** Define number of "out of tolerance" elements to tolerate
 *
 * Similar to @ref arm_compute::test::validation::relative_tolerance()
 *
 * @return the number of elements to tolerate
 */
float tolerance_num(DataType data_type, ActivationLayerInfo::ActivationFunction activation);

/** Calculate a suitable output quantization given the activation function and data type
 *
 * @param[in] data_type     The data type used.
 * @param[in] act_info      The activation function and additional information used.
 * @param[in] default_qinfo Default quantization info to be used.
 *
 * @return Output quantization info
 */
QuantizationInfo calculate_output_quantization_info(DataType                   data_type,
                                                    const ActivationLayerInfo &act_info,
                                                    const QuantizationInfo    &default_qinfo);

/** This function will return a vector filled with the following values that can
 *  represent two partitions derived from equivalent partitioning.
 *   - Lower parition: min, min + delta, lower quarter (nominal), center - delta
 *   - Upper partition: center, center + delta, upper quarter (nominal), max - delta, max
 *
 * @param[in] data_type The data type used.
 * @param[in] min       Minimum value to be used.
 * @param[in] max       Maximum value to be used.
 *
 * @return A vector of values of type T
 */
template <typename T>
std::vector<T> get_boundary_values(DataType data_type, T min, T max);

/** Define absolute tolerance of the activation layer for qasymm8.
 *
 * @param[in] activation The activation function used.
 *
 * @return Absolute tolerance depending on the activation function.
 */
AbsoluteTolerance<uint8_t> tolerance_qasymm8(ActivationLayerInfo::ActivationFunction activation);
} // namespace helper
} // namespace validation
} // namespace test
} // namespace arm_compute

#endif // ACL_TESTS_VALIDATION_HELPERS_ACTIVATIONHELPERS_H
