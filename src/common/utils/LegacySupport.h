/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef SRC_COMMON_LEGACY_SUPPORT_H
#define SRC_COMMON_LEGACY_SUPPORT_H

#include "arm_compute/Acl.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace detail
{
/** Convert a descriptor to a legacy format one
 *
 * @param[in] desc Descriptor to convert
 *
 * @return Legacy tensor meta-data
 */
TensorInfo convert_to_legacy_tensor_info(const AclTensorDescriptor &desc);
/** Convert a legacy tensor meta-data to a descriptor
 *
 * @param[in] info Legacy tensor meta-data
 *
 * @return A converted descriptor
 */
AclTensorDescriptor convert_to_descriptor(const TensorInfo &info);
/** Convert an AclActivation descriptor to an internal one
 *
 * @param[in] desc Descriptor to convert
 *
 * @return Legacy tensor meta-data
 */
ActivationLayerInfo convert_to_activation_info(const AclActivationDescriptor &desc);
} // namespace detail
} // namespace arm_compute

#endif /* SRC_COMMON_LEGACY_SUPPORT_H */
