/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_MISC_INFO_HELPERS_H__
#define __ARM_COMPUTE_MISC_INFO_HELPERS_H__

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace utils
{
namespace info_helpers
{
/** Checks if activation information correspond to a relu activation function
 *
 * @param[in] activation_info Activation metadata
 *
 * @return True if activation metadata correspond to a relu activation else false
 */
inline bool is_relu(ActivationLayerInfo activation_info)
{
    return activation_info.enabled() && activation_info.activation() == ActivationLayerInfo::ActivationFunction::RELU;
}

/** Checks if activation information correspond to a relu6 activation function
 *
 * @param[in] activation_info Activation metadata
 *
 * @return True if activation metadata correspond to a relu6 activation else false
 */
inline bool is_relu6(ActivationLayerInfo activation_info)
{
    const bool is_lu_bounded_relu = activation_info.activation() == ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU
                                    && activation_info.a() == 6.f && activation_info.b() == 0.f;
    const bool is_bounded_relu = activation_info.activation() == ActivationLayerInfo::ActivationFunction::BOUNDED_RELU
                                 && activation_info.a() == 6.f;
    return activation_info.enabled() && (is_lu_bounded_relu || is_bounded_relu);
}
} // namespace info_helpers
} // namespace utils
} // namespace arm_compute
#endif /* __ARM_COMPUTE_MISC_INFO_HELPERS_H__ */
