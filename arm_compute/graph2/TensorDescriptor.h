/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_GRAPH2_TENSOR_DESCRIPTOR_H__
#define __ARM_COMPUTE_GRAPH2_TENSOR_DESCRIPTOR_H__

#include "arm_compute/graph2/Types.h"

namespace arm_compute
{
namespace graph2
{
/** Tensor metadata class */
struct TensorDescriptor final
{
    /** Default Constructor **/
    TensorDescriptor() = default;
    /** Constructor
     *
     * @param[in] tensor_shape     Tensor shape
     * @param[in] tensor_data_type Tensor data type
     * @param[in] tensor_target    Target to allocate the tensor for
     */
    TensorDescriptor(TensorShape tensor_shape, DataType tensor_data_type, Target tensor_target = Target::UNSPECIFIED)
        : shape(tensor_shape), data_type(tensor_data_type), target(tensor_target)
    {
    }

    TensorShape shape{};                        /**< Tensor shape */
    DataType    data_type{ DataType::UNKNOWN }; /**< Data type */
    Target      target{ Target::UNSPECIFIED };  /**< Target */
};
} // namespace graph2
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH2_TENSOR_DESCRIPTOR_H__ */
