/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_LAYER_DESCRIPTORS_H
#define ARM_COMPUTE_LAYER_DESCRIPTORS_H

#include "arm_compute/core/Types.h"
#include "arm_compute/graph/Types.h"

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

/** Elementwise layer descriptor */
struct EltwiseLayerDescriptor
{
    /** Constructor
     *
     * @param[in] op               Element-wise operation to perform
     * @param[in] out_quant_info   (Optional) Output quantization information. Defaults to empty @ref QuantizationInfo
     * @param[in] c_policy         (Optional) Convert policy used for the operation. Defaults to @ref ConvertPolicy::SATURATE
     * @param[in] r_policy         (Optional) Rounding policy used for the operation. Defaults to @ref RoundingPolicy::TO_ZERO
     * @param[in] fused_activation (Optional) Fused activation information. Defaults to empty (identity) @ref ActivationLayerInfo
     */
    EltwiseLayerDescriptor(EltwiseOperation op, QuantizationInfo out_quant_info = QuantizationInfo(), ConvertPolicy c_policy = ConvertPolicy::SATURATE, RoundingPolicy r_policy = RoundingPolicy::TO_ZERO,
                           ActivationLayerInfo fused_activation = ActivationLayerInfo())
        : op(op), out_quant_info(out_quant_info), c_policy(c_policy), r_policy(r_policy), fused_activation(fused_activation)
    {
    }

    EltwiseOperation    op;               /**< Element-wise operation to perform */
    QuantizationInfo    out_quant_info;   /**< Output quantization information */
    ConvertPolicy       c_policy;         /**< Convert policy */
    RoundingPolicy      r_policy;         /**< Rounding policy */
    ActivationLayerInfo fused_activation; /**< Fused activation info */
};

/** Unary Elementwise layer descriptor */
struct UnaryEltwiseLayerDescriptor
{
    /** Constructor
     *
     * @param[in] op               Unary element-wise operation to perform
     * @param[in] out_quant_info   (Optional) Output quantization information. Defaults to empty @ref QuantizationInfo
     * @param[in] c_policy         (Optional) Convert policy used for the operation. Defaults to @ref ConvertPolicy::SATURATE
     * @param[in] r_policy         (Optional) Rounding policy used for the operation. Defaults to @ref RoundingPolicy::TO_ZERO
     * @param[in] fused_activation (Optional) Fused activation information. Defaults to empty (identity) @ref ActivationLayerInfo
     */
    UnaryEltwiseLayerDescriptor(UnaryEltwiseOperation op, QuantizationInfo out_quant_info = QuantizationInfo(), ConvertPolicy c_policy = ConvertPolicy::SATURATE,
                                RoundingPolicy      r_policy         = RoundingPolicy::TO_ZERO,
                                ActivationLayerInfo fused_activation = ActivationLayerInfo())
        : op(op), out_quant_info(out_quant_info), c_policy(c_policy), r_policy(r_policy), fused_activation(fused_activation)
    {
    }

    UnaryEltwiseOperation op;               /**< Unary element-wise operation to perform */
    QuantizationInfo      out_quant_info;   /**< Output quantization information */
    ConvertPolicy         c_policy;         /**< Convert policy */
    RoundingPolicy        r_policy;         /**< Rounding policy */
    ActivationLayerInfo   fused_activation; /**< Fused activation info */
};

/** Deconvolution layer descriptor */
struct DeconvolutionLayerDescriptor
{
    /** Constructor
     *
     * @param[in] info           Dedonvolution layer attributes
     * @param[in] out_quant_info (Optional) Output quantization infomation
     */
    DeconvolutionLayerDescriptor(PadStrideInfo info, QuantizationInfo out_quant_info = QuantizationInfo())
        : info(info), out_quant_info(out_quant_info)
    {
    }

    PadStrideInfo    info;           /**< Padding and stride information */
    QuantizationInfo out_quant_info; /**< Output quantization information */
};
} // namespace descriptor
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_LAYER_DESCRIPTORS_H */