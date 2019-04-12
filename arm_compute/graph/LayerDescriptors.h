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
#ifndef __ARM_COMPUTE_CONCAT_DESCRIPTOR_H__
#define __ARM_COMPUTE_CONCAT_DESCRIPTOR_H__

#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace graph
{
namespace descriptors
{
/** Concatenate layer descriptor */
struct ConcatLayerDescriptor
{
    /** Default constructor */
    ConcatLayerDescriptor()
        : axis(DataLayoutDimension::CHANNEL), output_qinfo()
    {
    }

    /** Constructor concatenate layer descriptor
     *
     * @param[in] axis Axis.
     */
    ConcatLayerDescriptor(DataLayoutDimension axis)
        : axis(axis), output_qinfo()
    {
    }

    /** Constructor concatenate layer descriptor
     *
     * @param[in] axis         Axis.
     * @param[in] output_qinfo Output quantization info.
     */
    ConcatLayerDescriptor(DataLayoutDimension axis, QuantizationInfo output_qinfo)
        : axis(axis), output_qinfo(output_qinfo)
    {
    }

    const DataLayoutDimension axis;         /**< Concatenation Axis */
    const QuantizationInfo    output_qinfo; /**< Output quantizazion info */
};
} // namespace descriptor
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CONCAT_DESCRIPTOR_H__ */