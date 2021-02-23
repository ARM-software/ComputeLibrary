/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_GRAPH_TENSOR_DESCRIPTOR_H
#define ARM_COMPUTE_GRAPH_TENSOR_DESCRIPTOR_H

#include "arm_compute/graph/Types.h"

#include "support/ICloneable.h"

#include <memory>

namespace arm_compute
{
namespace graph
{
/** Tensor metadata class */
struct TensorDescriptor final : public misc::ICloneable<TensorDescriptor>
{
    /** Default Constructor **/
    TensorDescriptor() = default;
    /** Constructor
     *
     * @param[in] tensor_shape       Tensor shape
     * @param[in] tensor_data_type   Tensor data type
     * @param[in] tensor_quant_info  Tensor quantization info
     * @param[in] tensor_data_layout Tensor data layout
     * @param[in] tensor_target      Target to allocate the tensor for
     */
    TensorDescriptor(TensorShape      tensor_shape,
                     DataType         tensor_data_type,
                     QuantizationInfo tensor_quant_info  = QuantizationInfo(),
                     DataLayout       tensor_data_layout = DataLayout::NCHW,
                     Target           tensor_target      = Target::UNSPECIFIED)
        : shape(tensor_shape), data_type(tensor_data_type), layout(tensor_data_layout), quant_info(tensor_quant_info), target(tensor_target)
    {
    }
    /** Sets tensor descriptor shape
     *
     * @param[in] tensor_shape Tensor shape to set
     *
     * @return This tensor descriptor
     */
    TensorDescriptor &set_shape(TensorShape &tensor_shape)
    {
        shape = tensor_shape;
        return *this;
    }
    /** Sets tensor descriptor data type
     *
     * @param[in] tensor_data_type Data type
     *
     * @return This tensor descriptor
     */
    TensorDescriptor &set_data_type(DataType tensor_data_type)
    {
        data_type = tensor_data_type;
        return *this;
    }
    /** Sets tensor descriptor data layout
     *
     * @param[in] data_layout Data layout
     *
     * @return This tensor descriptor
     */
    TensorDescriptor &set_layout(DataLayout data_layout)
    {
        layout = data_layout;
        return *this;
    }
    /** Sets tensor descriptor quantization info
     *
     * @param[in] tensor_quant_info Quantization information
     *
     * @return This tensor descriptor
     */
    TensorDescriptor &set_quantization_info(QuantizationInfo tensor_quant_info)
    {
        quant_info = tensor_quant_info;
        return *this;
    }

    // Inherited methods overridden:
    std::unique_ptr<TensorDescriptor> clone() const override
    {
        return std::make_unique<TensorDescriptor>(*this);
    }

    TensorShape      shape{};                        /**< Tensor shape */
    DataType         data_type{ DataType::UNKNOWN }; /**< Data type */
    DataLayout       layout{ DataLayout::NCHW };     /**< Data layout */
    QuantizationInfo quant_info{};                   /**< Quantization info */
    Target           target{ Target::UNSPECIFIED };  /**< Target */
};
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_TENSOR_DESCRIPTOR_H */
