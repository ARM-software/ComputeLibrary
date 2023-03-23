/*
 * Copyright (c) 2017-2023 Arm Limited.
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
#ifndef __ARM_COMPUTE_TYPE_PRINTER_H__
#define __ARM_COMPUTE_TYPE_PRINTER_H__

#ifdef ARM_COMPUTE_OPENCL_ENABLED
#include "arm_compute/core/CL/ICLTensor.h"
#endif /* ARM_COMPUTE_OPENCL_ENABLED */

#include "arm_compute/core/Dimensions.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Strides.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/experimental/IPostOp.h"
#include "arm_compute/core/experimental/PostOps.h"
#include "arm_compute/dynamic_fusion/sketch/attributes/CastAttributes.h"
#include "arm_compute/dynamic_fusion/sketch/attributes/ClampAttributes.h"
#include "arm_compute/dynamic_fusion/sketch/attributes/Conv2dAttributes.h"
#include "arm_compute/dynamic_fusion/sketch/attributes/DepthwiseConv2dAttributes.h"
#include "arm_compute/dynamic_fusion/sketch/attributes/Pool2dAttributes.h"
#include "arm_compute/dynamic_fusion/sketch/attributes/ResizeAttributes.h"
#include "arm_compute/dynamic_fusion/sketch/attributes/SoftmaxAttributes.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuPool2d.h"
#include "arm_compute/runtime/CL/CLTunerTypes.h"
#include "arm_compute/runtime/CL/CLTypes.h"
#include "arm_compute/runtime/FunctionDescriptors.h"
#include "arm_compute/runtime/NEON/functions/NEMatMul.h"
#include "arm_compute/runtime/common/LSTMParams.h"
#include "support/Cast.h"
#include "support/StringSupport.h"
#include <ostream>
#include <sstream>
#include <string>

namespace arm_compute
{
/** Formatted output if arg is not null
 *
 * @param[in] arg Object to print
 *
 * @return String representing arg.
 */
template <typename T>
std::string to_string_if_not_null(T *arg)
{
    if(arg == nullptr)
    {
        return "nullptr";
    }
    else
    {
        return to_string(*arg);
    }
}

/** Fallback method: try to use std::to_string:
 *
 * @param[in] val Value to convert to string
 *
 * @return String representing val.
 */
template <typename T>
inline std::string to_string(const T &val)
{
    return support::cpp11::to_string(val);
}

/** Formatted output of a vector of objects.
 *
 * @note: Using the overloaded to_string() instead of overloaded operator<<(), because to_string() functions are
 *        overloaded for all types, where two or more of them can use the same operator<<(), ITensor is an example.
 *
 * @param[out] os   Output stream
 * @param[in]  args Vector of objects to print
 *
 * @return Modified output stream.
 */
template <typename T>
::std::ostream &operator<<(::std::ostream &os, const std::vector<T> &args)
{
    const size_t max_print_size = 5U;

    os << "[";
    bool   first = true;
    size_t i;
    for(i = 0; i < args.size(); ++i)
    {
        if(i == max_print_size)
        {
            break;
        }
        if(first)
        {
            first = false;
        }
        else
        {
            os << ", ";
        }
        os << to_string(args[i]);
    }
    if(i < args.size())
    {
        os << ", ...";
    }
    os << "]";
    return os;
}

/** Formatted output of a vector of objects.
 *
 * @param[in] args Vector of objects to print
 *
 * @return String representing args.
 */
template <typename T>
std::string to_string(const std::vector<T> &args)
{
    std::stringstream str;
    str << args;
    return str.str();
}

/** @name (EXPERIMENTAL_POST_OPS)
 * @{
 */
/** Formmated output of the @ref experimental::PostOpType type
 *
 * @param[out] os           Output stream.
 * @param[in]  post_op_type Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, experimental::PostOpType post_op_type)
{
    os << "type=";
    switch(post_op_type)
    {
        case experimental::PostOpType::Activation:
        {
            os << "Activation";
            break;
        }
        case experimental::PostOpType::Eltwise_Add:
        {
            os << "Eltwise_Add";
            break;
        }
        case experimental::PostOpType::Eltwise_PRelu:
        {
            os << "Eltwise_PRelu";
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Unsupported PostOpType");
            break;
        }
    }
    return os;
}
/** Converts a @ref experimental::PostOpType to string
 *
 * @param[in] post_op_type PostOpType value to be converted
 *
 * @return String representing the corresponding PostOpType
 */
inline std::string to_string(experimental::PostOpType post_op_type)
{
    std::stringstream str;
    str << post_op_type;
    return str.str();
}
/** Formatted output of the @ref experimental::IPostOp type.
 *
 * @param[out] os      Output stream.
 * @param[in]  post_op Type to output.
 *
 * @return Modified output stream.
 */
template <typename T>
inline ::std::ostream &operator<<(::std::ostream &os, const experimental::IPostOp<T> &post_op)
{
    os << "<";
    os << post_op.type() << ",";
    os << "prev_dst_pos=" << post_op.prev_dst_pos() << ",";
    switch(post_op.type())
    {
        case experimental::PostOpType::Activation:
        {
            const auto _post_op = utils::cast::polymorphic_downcast<const experimental::PostOpAct<T> *>(&post_op);
            os << "act_info=" << &(_post_op->_act_info);
            break;
        }
        case experimental::PostOpType::Eltwise_Add:
        {
            const auto _post_op = utils::cast::polymorphic_downcast<const experimental::PostOpEltwiseAdd<T> *>(&post_op);
            os << "convert_policy=" << _post_op->_policy;
            break;
        }
        case experimental::PostOpType::Eltwise_PRelu:
        {
            const auto _post_op = utils::cast::polymorphic_downcast<const experimental::PostOpEltwisePRelu<T> *>(&post_op);
            os << "convert_policy=" << _post_op->_policy;
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Unsupported PostOpType");
            break;
        }
    }
    os << ">";
    return os;
}
/** Converts an @ref experimental::IPostOp to string
 *
 * @param[in] post_op IPostOp value to be converted
 *
 * @return String representing the corresponding IPostOp
 */
template <typename T>
inline std::string to_string(const experimental::IPostOp<T> &post_op)
{
    std::stringstream str;
    str << post_op;
    return str.str();
}
/** Formatted output of the @ref experimental::PostOpList type.
 *
 * @param[out] os       Output stream.
 * @param[in]  post_ops Type to output.
 *
 * @return Modified output stream.
 */
template <typename T>
inline ::std::ostream &operator<<(::std::ostream &os, const experimental::PostOpList<T> &post_ops)
{
    os << "[";
    for(const auto &post_op : post_ops.get_list())
    {
        os << *post_op << ",";
    }
    os << "]";
    return os;
}
/** Converts a @ref experimental::PostOpList to string
 *
 * @param[in] post_ops PostOpList value to be converted
 *
 * @return String representing the corresponding PostOpList
 */
template <typename T>
inline std::string to_string(const experimental::PostOpList<T> &post_ops)
{
    std::stringstream str;
    str << post_ops;
    return str.str();
}
/** @} */ // end of group (EXPERIMENTAL_POST_OPS)

/** Formatted output of the Dimensions type.
 *
 * @param[out] os         Output stream.
 * @param[in]  dimensions Type to output.
 *
 * @return Modified output stream.
 */
template <typename T>
inline ::std::ostream &operator<<(::std::ostream &os, const Dimensions<T> &dimensions)
{
    if(dimensions.num_dimensions() > 0)
    {
        os << dimensions[0];

        for(unsigned int d = 1; d < dimensions.num_dimensions(); ++d)
        {
            os << "," << dimensions[d];
        }
    }

    return os;
}

/** Formatted output of the RoundingPolicy type.
 *
 * @param[out] os              Output stream.
 * @param[in]  rounding_policy Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const RoundingPolicy &rounding_policy)
{
    switch(rounding_policy)
    {
        case RoundingPolicy::TO_ZERO:
            os << "TO_ZERO";
            break;
        case RoundingPolicy::TO_NEAREST_UP:
            os << "TO_NEAREST_UP";
            break;
        case RoundingPolicy::TO_NEAREST_EVEN:
            os << "TO_NEAREST_EVEN";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the WeightsInfo type.
 *
 * @param[out] os           Output stream.
 * @param[in]  weights_info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const WeightsInfo &weights_info)
{
    os << weights_info.are_reshaped() << ";";
    os << weights_info.num_kernels() << ";" << weights_info.kernel_size().first << "," << weights_info.kernel_size().second;

    return os;
}

/** Formatted output of the ROIPoolingInfo type.
 *
 * @param[out] os        Output stream.
 * @param[in]  pool_info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const ROIPoolingLayerInfo &pool_info)
{
    os << pool_info.pooled_width() << "x" << pool_info.pooled_height() << "~" << pool_info.spatial_scale();
    return os;
}

/** Formatted output of the ROIPoolingInfo type.
 *
 * @param[in] pool_info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const ROIPoolingLayerInfo &pool_info)
{
    std::stringstream str;
    str << pool_info;
    return str.str();
}

/** Formatted output of the GEMMKernelInfo type.
 *
 * @param[out] os        Output stream.
 * @param[in]  gemm_info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const GEMMKernelInfo &gemm_info)
{
    os << "( m=" << gemm_info.m;
    os << " n=" << gemm_info.n;
    os << " k=" << gemm_info.k;
    os << " depth_output_gemm3d=" << gemm_info.depth_output_gemm3d;
    os << " reinterpret_input_as_3d=" << gemm_info.reinterpret_input_as_3d;
    os << " broadcast_bias=" << gemm_info.broadcast_bias;
    os << " fp_mixed_precision=" << gemm_info.fp_mixed_precision;
    os << " mult_transpose1xW_width=" << gemm_info.mult_transpose1xW_width;
    os << " mult_interleave4x4_height=" << gemm_info.mult_interleave4x4_height;
    os << " a_offset=" << gemm_info.a_offset;
    os << " b_offset=" << gemm_info.b_offset;
    os << "post_ops=" << gemm_info.post_ops;
    os << ")";
    return os;
}

/** Formatted output of the GEMMLHSMatrixInfo type.
 *
 * @param[out] os        Output stream.
 * @param[in]  gemm_info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const GEMMLHSMatrixInfo &gemm_info)
{
    os << "( m0=" << (unsigned int)gemm_info.m0 << " k0=" << gemm_info.k0 << "  v0=" << gemm_info.v0 << "  trans=" << gemm_info.transpose << "  inter=" << gemm_info.interleave << "})";
    return os;
}

/** Formatted output of the GEMMRHSMatrixInfo type.
 *
 * @param[out] os        Output stream.
 * @param[in]  gemm_info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const GEMMRHSMatrixInfo &gemm_info)
{
    os << "( n0=" << (unsigned int)gemm_info.n0 << " k0=" << gemm_info.k0 << "  h0=" << gemm_info.h0 << "  trans=" << gemm_info.transpose << "  inter=" << gemm_info.interleave << " exp_img=" <<
       gemm_info.export_to_cl_image << "})";
    return os;
}

/** Formatted output of the GEMMRHSMatrixInfo type.
 *
 * @param[in] gemm_info GEMMRHSMatrixInfo to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const GEMMRHSMatrixInfo &gemm_info)
{
    std::stringstream str;
    str << gemm_info;
    return str.str();
}

/** Formatted output of the GEMMLHSMatrixInfo type.
 *
 * @param[in] gemm_info GEMMLHSMatrixInfo to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const GEMMLHSMatrixInfo &gemm_info)
{
    std::stringstream str;
    str << gemm_info;
    return str.str();
}

/** Formatted output of the GEMMKernelInfo type.
 *
 * @param[in] gemm_info GEMMKernelInfo Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const GEMMKernelInfo &gemm_info)
{
    std::stringstream str;
    str << gemm_info;
    return str.str();
}

/** Formatted output of the BoundingBoxTransformInfo type.
 *
 * @param[out] os        Output stream.
 * @param[in]  bbox_info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const BoundingBoxTransformInfo &bbox_info)
{
    auto weights = bbox_info.weights();
    os << "(" << bbox_info.img_width() << "x" << bbox_info.img_height() << ")~" << bbox_info.scale() << "(weights={" << weights[0] << ", " << weights[1] << ", " << weights[2] << ", " << weights[3] <<
       "})";
    return os;
}

#if defined(ARM_COMPUTE_ENABLE_BF16)
inline ::std::ostream &operator<<(::std::ostream &os, const bfloat16 &v)
{
    std::stringstream str;
    str << v;
    os << str.str();
    return os;
}
#endif /* defined(ARM_COMPUTE_ENABLE_BF16) */

/** Formatted output of the BoundingBoxTransformInfo type.
 *
 * @param[in] bbox_info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const BoundingBoxTransformInfo &bbox_info)
{
    std::stringstream str;
    str << bbox_info;
    return str.str();
}

/** Formatted output of the ComputeAnchorsInfo type.
 *
 * @param[out] os           Output stream.
 * @param[in]  anchors_info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const ComputeAnchorsInfo &anchors_info)
{
    os << "(" << anchors_info.feat_width() << "x" << anchors_info.feat_height() << ")~" << anchors_info.spatial_scale();
    return os;
}

/** Formatted output of the ComputeAnchorsInfo type.
 *
 * @param[in] anchors_info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const ComputeAnchorsInfo &anchors_info)
{
    std::stringstream str;
    str << anchors_info;
    return str.str();
}

/** Formatted output of the GenerateProposalsInfo type.
 *
 * @param[out] os             Output stream.
 * @param[in]  proposals_info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const GenerateProposalsInfo &proposals_info)
{
    os << "(" << proposals_info.im_width() << "x" << proposals_info.im_height() << ")~" << proposals_info.im_scale();
    return os;
}

/** Formatted output of the GenerateProposalsInfo type.
 *
 * @param[in] proposals_info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const GenerateProposalsInfo &proposals_info)
{
    std::stringstream str;
    str << proposals_info;
    return str.str();
}

/** Formatted output of the QuantizationInfo type.
 *
 * @param[out] os    Output stream.
 * @param[in]  qinfo Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const QuantizationInfo &qinfo)
{
    const UniformQuantizationInfo uqinfo = qinfo.uniform();
    os << "Scale:" << uqinfo.scale << "~";
    os << "Offset:" << uqinfo.offset;
    return os;
}

/** Formatted output of the QuantizationInfo type.
 *
 * @param[in] quantization_info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const QuantizationInfo &quantization_info)
{
    std::stringstream str;
    str << quantization_info;
    return str.str();
}

/** Formatted output of the activation function type.
 *
 * @param[out] os           Output stream.
 * @param[in]  act_function Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const ActivationLayerInfo::ActivationFunction &act_function)
{
    switch(act_function)
    {
        case ActivationLayerInfo::ActivationFunction::ABS:
            os << "ABS";
            break;
        case ActivationLayerInfo::ActivationFunction::LINEAR:
            os << "LINEAR";
            break;
        case ActivationLayerInfo::ActivationFunction::LOGISTIC:
            os << "LOGISTIC";
            break;
        case ActivationLayerInfo::ActivationFunction::RELU:
            os << "RELU";
            break;
        case ActivationLayerInfo::ActivationFunction::BOUNDED_RELU:
            os << "BOUNDED_RELU";
            break;
        case ActivationLayerInfo::ActivationFunction::LEAKY_RELU:
            os << "LEAKY_RELU";
            break;
        case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
            os << "SOFT_RELU";
            break;
        case ActivationLayerInfo::ActivationFunction::SQRT:
            os << "SQRT";
            break;
        case ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU:
            os << "LU_BOUNDED_RELU";
            break;
        case ActivationLayerInfo::ActivationFunction::ELU:
            os << "ELU";
            break;
        case ActivationLayerInfo::ActivationFunction::SQUARE:
            os << "SQUARE";
            break;
        case ActivationLayerInfo::ActivationFunction::TANH:
            os << "TANH";
            break;
        case ActivationLayerInfo::ActivationFunction::IDENTITY:
            os << "IDENTITY";
            break;
        case ActivationLayerInfo::ActivationFunction::HARD_SWISH:
            os << "HARD_SWISH";
            break;
        case ActivationLayerInfo::ActivationFunction::SWISH:
            os << "SWISH";
            break;
        case ActivationLayerInfo::ActivationFunction::GELU:
            os << "GELU";
            break;

        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the activation function info type.
 *
 * @param[in] info ActivationLayerInfo to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const arm_compute::ActivationLayerInfo &info)
{
    std::stringstream str;
    if(info.enabled())
    {
        str << info.activation();
    }
    return str.str();
}

/** Formatted output of the activation function info.
 *
 * @param[out] os   Output stream.
 * @param[in]  info ActivationLayerInfo to output.
 *
 * @return Formatted string.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const ActivationLayerInfo *info)
{
    if(info != nullptr)
    {
        if(info->enabled())
        {
            os << info->activation();
            os << "(";
            os << "VAL_A=" << info->a() << ",";
            os << "VAL_B=" << info->b();
            os << ")";
        }
        else
        {
            os << "disabled";
        }
    }
    else
    {
        os << "nullptr";
    }
    return os;
}

/** Formatted output of the activation function type.
 *
 * @param[in] function Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const arm_compute::ActivationLayerInfo::ActivationFunction &function)
{
    std::stringstream str;
    str << function;
    return str.str();
}

/** Formatted output of the NormType type.
 *
 * @param[out] os        Output stream.
 * @param[in]  norm_type Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const NormType &norm_type)
{
    switch(norm_type)
    {
        case NormType::CROSS_MAP:
            os << "CROSS_MAP";
            break;
        case NormType::IN_MAP_1D:
            os << "IN_MAP_1D";
            break;
        case NormType::IN_MAP_2D:
            os << "IN_MAP_2D";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of @ref NormalizationLayerInfo.
 *
 * @param[in] info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const arm_compute::NormalizationLayerInfo &info)
{
    std::stringstream str;
    str << info.type() << ":NormSize=" << info.norm_size();
    return str.str();
}

/** Formatted output of @ref NormalizationLayerInfo.
 *
 * @param[out] os   Output stream.
 * @param[in]  info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const NormalizationLayerInfo &info)
{
    os << info.type() << ":NormSize=" << info.norm_size();
    return os;
}

/** Formatted output of the PoolingType type.
 *
 * @param[out] os        Output stream.
 * @param[in]  pool_type Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const PoolingType &pool_type)
{
    switch(pool_type)
    {
        case PoolingType::AVG:
            os << "AVG";
            break;
        case PoolingType::MAX:
            os << "MAX";
            break;
        case PoolingType::L2:
            os << "L2";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of @ref PoolingLayerInfo.
 *
 * @param[out] os   Output stream.
 * @param[in]  info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const PoolingLayerInfo &info)
{
    os << info.pool_type;

    return os;
}

/** Formatted output of @ref RoundingPolicy.
 *
 * @param[in] rounding_policy Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const RoundingPolicy &rounding_policy)
{
    std::stringstream str;
    str << rounding_policy;
    return str.str();
}

/** [Print DataLayout type] **/
/** Formatted output of the DataLayout type.
 *
 * @param[out] os          Output stream.
 * @param[in]  data_layout Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const DataLayout &data_layout)
{
    switch(data_layout)
    {
        case DataLayout::UNKNOWN:
            os << "UNKNOWN";
            break;
        case DataLayout::NHWC:
            os << "NHWC";
            break;
        case DataLayout::NCHW:
            os << "NCHW";
            break;
        case DataLayout::NDHWC:
            os << "NDHWC";
            break;
        case DataLayout::NCDHW:
            os << "NCDHW";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the DataLayout type.
 *
 * @param[in] data_layout Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const arm_compute::DataLayout &data_layout)
{
    std::stringstream str;
    str << data_layout;
    return str.str();
}
/** [Print DataLayout type] **/

/** Formatted output of the DataLayoutDimension type.
 *
 * @param[out] os              Output stream.
 * @param[in]  data_layout_dim Data layout dimension to print.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const DataLayoutDimension &data_layout_dim)
{
    switch(data_layout_dim)
    {
        case DataLayoutDimension::WIDTH:
            os << "WIDTH";
            break;
        case DataLayoutDimension::HEIGHT:
            os << "HEIGHT";
            break;
        case DataLayoutDimension::CHANNEL:
            os << "CHANNEL";
            break;
        case DataLayoutDimension::DEPTH:
            os << "DEPTH";
            break;
        case DataLayoutDimension::BATCHES:
            os << "BATCHES";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }
    return os;
}

/** Formatted output of the DataType type.
 *
 * @param[out] os        Output stream.
 * @param[in]  data_type Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const DataType &data_type)
{
    switch(data_type)
    {
        case DataType::UNKNOWN:
            os << "UNKNOWN";
            break;
        case DataType::U8:
            os << "U8";
            break;
        case DataType::QSYMM8:
            os << "QSYMM8";
            break;
        case DataType::QASYMM8:
            os << "QASYMM8";
            break;
        case DataType::QASYMM8_SIGNED:
            os << "QASYMM8_SIGNED";
            break;
        case DataType::QSYMM8_PER_CHANNEL:
            os << "QSYMM8_PER_CHANNEL";
            break;
        case DataType::S8:
            os << "S8";
            break;
        case DataType::U16:
            os << "U16";
            break;
        case DataType::S16:
            os << "S16";
            break;
        case DataType::QSYMM16:
            os << "QSYMM16";
            break;
        case DataType::QASYMM16:
            os << "QASYMM16";
            break;
        case DataType::U32:
            os << "U32";
            break;
        case DataType::S32:
            os << "S32";
            break;
        case DataType::U64:
            os << "U64";
            break;
        case DataType::S64:
            os << "S64";
            break;
        case DataType::BFLOAT16:
            os << "BFLOAT16";
            break;
        case DataType::F16:
            os << "F16";
            break;
        case DataType::F32:
            os << "F32";
            break;
        case DataType::F64:
            os << "F64";
            break;
        case DataType::SIZET:
            os << "SIZET";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the DataType type.
 *
 * @param[in] data_type Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const arm_compute::DataType &data_type)
{
    std::stringstream str;
    str << data_type;
    return str.str();
}

/** Formatted output of the Format type.
 *
 * @param[out] os     Output stream.
 * @param[in]  format Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const Format &format)
{
    switch(format)
    {
        case Format::UNKNOWN:
            os << "UNKNOWN";
            break;
        case Format::U8:
            os << "U8";
            break;
        case Format::S16:
            os << "S16";
            break;
        case Format::U16:
            os << "U16";
            break;
        case Format::S32:
            os << "S32";
            break;
        case Format::U32:
            os << "U32";
            break;
        case Format::F16:
            os << "F16";
            break;
        case Format::F32:
            os << "F32";
            break;
        case Format::UV88:
            os << "UV88";
            break;
        case Format::RGB888:
            os << "RGB888";
            break;
        case Format::RGBA8888:
            os << "RGBA8888";
            break;
        case Format::YUV444:
            os << "YUV444";
            break;
        case Format::YUYV422:
            os << "YUYV422";
            break;
        case Format::NV12:
            os << "NV12";
            break;
        case Format::NV21:
            os << "NV21";
            break;
        case Format::IYUV:
            os << "IYUV";
            break;
        case Format::UYVY422:
            os << "UYVY422";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the Format type.
 *
 * @param[in] format Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const Format &format)
{
    std::stringstream str;
    str << format;
    return str.str();
}

/** Formatted output of the Channel type.
 *
 * @param[out] os      Output stream.
 * @param[in]  channel Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const Channel &channel)
{
    switch(channel)
    {
        case Channel::UNKNOWN:
            os << "UNKNOWN";
            break;
        case Channel::C0:
            os << "C0";
            break;
        case Channel::C1:
            os << "C1";
            break;
        case Channel::C2:
            os << "C2";
            break;
        case Channel::C3:
            os << "C3";
            break;
        case Channel::R:
            os << "R";
            break;
        case Channel::G:
            os << "G";
            break;
        case Channel::B:
            os << "B";
            break;
        case Channel::A:
            os << "A";
            break;
        case Channel::Y:
            os << "Y";
            break;
        case Channel::U:
            os << "U";
            break;
        case Channel::V:
            os << "V";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the Channel type.
 *
 * @param[in] channel Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const Channel &channel)
{
    std::stringstream str;
    str << channel;
    return str.str();
}

/** Formatted output of the BorderMode type.
 *
 * @param[out] os   Output stream.
 * @param[in]  mode Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const BorderMode &mode)
{
    switch(mode)
    {
        case BorderMode::UNDEFINED:
            os << "UNDEFINED";
            break;
        case BorderMode::CONSTANT:
            os << "CONSTANT";
            break;
        case BorderMode::REPLICATE:
            os << "REPLICATE";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the BorderSize type.
 *
 * @param[out] os     Output stream.
 * @param[in]  border Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const BorderSize &border)
{
    os << border.top << ","
       << border.right << ","
       << border.bottom << ","
       << border.left;

    return os;
}

/** Formatted output of the PaddingList type.
 *
 * @param[out] os      Output stream.
 * @param[in]  padding Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const PaddingList &padding)
{
    os << "{";
    for(auto const &p : padding)
    {
        os << "{" << p.first << "," << p.second << "}";
    }
    os << "}";
    return os;
}

/** Formatted output of the Multiples type.
 *
 * @param[out] os        Output stream.
 * @param[in]  multiples Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const Multiples &multiples)
{
    os << "(";
    for(size_t i = 0; i < multiples.size() - 1; i++)
    {
        os << multiples[i] << ", ";
    }
    os << multiples.back() << ")";
    return os;
}

/** Formatted output of the InterpolationPolicy type.
 *
 * @param[out] os     Output stream.
 * @param[in]  policy Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const InterpolationPolicy &policy)
{
    switch(policy)
    {
        case InterpolationPolicy::NEAREST_NEIGHBOR:
            os << "NEAREST_NEIGHBOR";
            break;
        case InterpolationPolicy::BILINEAR:
            os << "BILINEAR";
            break;
        case InterpolationPolicy::AREA:
            os << "AREA";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the SamplingPolicy type.
 *
 * @param[out] os     Output stream.
 * @param[in]  policy Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const SamplingPolicy &policy)
{
    switch(policy)
    {
        case SamplingPolicy::CENTER:
            os << "CENTER";
            break;
        case SamplingPolicy::TOP_LEFT:
            os << "TOP_LEFT";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the ITensorInfo type.
 *
 * @param[out] os   Output stream.
 * @param[in]  info Tensor information.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(std::ostream &os, const ITensorInfo *info)
{
    const DataType   data_type   = info->data_type();
    const DataLayout data_layout = info->data_layout();

    os << "Shape=" << info->tensor_shape() << ","
       << "DataLayout=" << string_from_data_layout(data_layout) << ","
       << "DataType=" << string_from_data_type(data_type);

    if(is_data_type_quantized(data_type))
    {
        const QuantizationInfo qinfo   = info->quantization_info();
        const auto             scales  = qinfo.scale();
        const auto             offsets = qinfo.offset();

        os << ", QuantizationInfo={"
           << "scales.size=" << scales.size()
           << ", scale(s)=" << scales << ", ";

        os << "offsets.size=" << offsets.size()
           << ", offset(s)=" << offsets << "}";
    }
    return os;
}

/** Formatted output of the const TensorInfo& type.
 *
 * @param[out] os   Output stream.
 * @param[in]  info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const TensorInfo &info)
{
    os << &info;
    return os;
}

/** Formatted output of the const TensorInfo& type.
 *
 * @param[in] info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const TensorInfo &info)
{
    std::stringstream str;
    str << &info;
    return str.str();
}

/** Formatted output of the const ITensorInfo& type.
 *
 * @param[in] info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const ITensorInfo &info)
{
    std::stringstream str;
    str << &info;
    return str.str();
}

/** Formatted output of the const ITensorInfo* type.
 *
 * @param[in] info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const ITensorInfo *info)
{
    std::string ret_str = "nullptr";
    if(info != nullptr)
    {
        std::stringstream str;
        str << info;
        ret_str = str.str();
    }
    return ret_str;
}

/** Formatted output of the ITensorInfo* type.
 *
 * @param[in] info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(ITensorInfo *info)
{
    return to_string(static_cast<const ITensorInfo *>(info));
}

/** Formatted output of the ITensorInfo type obtained from const ITensor* type.
 *
 * @param[in] tensor Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const ITensor *tensor)
{
    std::string ret_str = "nullptr";
    if(tensor != nullptr)
    {
        std::stringstream str;
        str << "ITensor->info(): " << tensor->info();
        ret_str = str.str();
    }
    return ret_str;
}

/** Formatted output of the ITensorInfo type obtained from the ITensor* type.
 *
 * @param[in] tensor Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(ITensor *tensor)
{
    return to_string(static_cast<const ITensor *>(tensor));
}

/** Formatted output of the ITensorInfo type obtained from the ITensor& type.
 *
 * @param[in] tensor Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(ITensor &tensor)
{
    std::stringstream str;
    str << "ITensor.info(): " << tensor.info();
    return str.str();
}

#ifdef ARM_COMPUTE_OPENCL_ENABLED
/** Formatted output of the ITensorInfo type obtained from the const ICLTensor& type.
 *
 * @param[in] cl_tensor Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const ICLTensor *cl_tensor)
{
    std::string ret_str = "nullptr";
    if(cl_tensor != nullptr)
    {
        std::stringstream str;
        str << "ICLTensor->info(): " << cl_tensor->info();
        ret_str = str.str();
    }
    return ret_str;
}

/** Formatted output of the ITensorInfo type obtained from the ICLTensor& type.
 *
 * @param[in] cl_tensor Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(ICLTensor *cl_tensor)
{
    return to_string(static_cast<const ICLTensor *>(cl_tensor));
}

/** Formatted output of the cl::NDRange type.
 *
 * @param[out] os       Output stream.
 * @param[in]  nd_range cl::NDRange to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const cl::NDRange &nd_range)
{
    os << "{"
       << nd_range[0] << ","
       << nd_range[1] << ","
       << nd_range[2]
       << "}";
    return os;
}

/** Formatted output of the cl::NDRange type
 *
 * @param[in] nd_Range Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const cl::NDRange &nd_range)
{
    std::stringstream str;
    str << nd_range;
    return str.str();
}
#endif /* ARM_COMPUTE_OPENCL_ENABLED */

/** Formatted output of the Dimensions type.
 *
 * @param[in] dimensions Type to output.
 *
 * @return Formatted string.
 */
template <typename T>
inline std::string to_string(const Dimensions<T> &dimensions)
{
    std::stringstream str;
    str << dimensions;
    return str.str();
}

/** Formatted output of the Strides type.
 *
 * @param[in] stride Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const Strides &stride)
{
    std::stringstream str;
    str << stride;
    return str.str();
}

/** Formatted output of the TensorShape type.
 *
 * @param[in] shape Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const TensorShape &shape)
{
    std::stringstream str;
    str << shape;
    return str.str();
}

/** Formatted output of the Coordinates type.
 *
 * @param[in] coord Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const Coordinates &coord)
{
    std::stringstream str;
    str << coord;
    return str.str();
}

/** Formatted output of the GEMMReshapeInfo type.
 *
 * @param[out] os   Output stream.
 * @param[in]  info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const GEMMReshapeInfo &info)
{
    os << "{m=" << info.m() << ",";
    os << "n=" << info.n() << ",";
    os << "k=" << info.k() << ",";
    os << "mult_transpose1xW_width=" << info.mult_transpose1xW_width() << ",";
    os << "mult_interleave4x4_height=" << info.mult_interleave4x4_height();
    os << "}";

    return os;
}

/** Formatted output of the GEMMInfo type.
 *
 * @param[out] os   Output stream.
 * @param[in]  info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const GEMMInfo &info)
{
    os << "{is_a_reshaped=" << info.is_a_reshaped() << ",";
    os << "is_b_reshaped=" << info.is_b_reshaped() << ",";
    os << "reshape_b_only_on_first_run=" << info.reshape_b_only_on_first_run() << ",";
    os << "depth_output_gemm3d=" << info.depth_output_gemm3d() << ",";
    os << "reinterpret_input_as_3d=" << info.reinterpret_input_as_3d() << ",";
    os << "retain_internal_weights=" << info.retain_internal_weights() << ",";
    os << "fp_mixed_precision=" << info.fp_mixed_precision() << ",";
    os << "broadcast_bias=" << info.broadcast_bias() << ",";
    os << "pretranspose_B=" << info.pretranspose_B() << ",";
    os << "post_ops=" << info.post_ops() << "}";

    return os;
}

/** Formatted output of the Window::Dimension type.
 *
 * @param[out] os  Output stream.
 * @param[in]  dim Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const Window::Dimension &dim)
{
    os << "{start=" << dim.start() << ", end=" << dim.end() << ", step=" << dim.step() << "}";

    return os;
}
/** Formatted output of the Window type.
 *
 * @param[out] os  Output stream.
 * @param[in]  win Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const Window &win)
{
    os << "{";
    for(unsigned int i = 0; i < Coordinates::num_max_dimensions; i++)
    {
        if(i > 0)
        {
            os << ", ";
        }
        os << win[i];
    }
    os << "}";

    return os;
}

/** Formatted output of the WeightsInfo type.
 *
 * @param[in] info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const WeightsInfo &info)
{
    std::stringstream str;
    str << info;
    return str.str();
}

/** Formatted output of the GEMMReshapeInfo type.
 *
 * @param[in] info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const GEMMReshapeInfo &info)
{
    std::stringstream str;
    str << info;
    return str.str();
}

/** Formatted output of the GEMMInfo type.
 *
 * @param[in] info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const GEMMInfo &info)
{
    std::stringstream str;
    str << info;
    return str.str();
}

/** Formatted output of the Window::Dimension type.
 *
 * @param[in] dim Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const Window::Dimension &dim)
{
    std::stringstream str;
    str << dim;
    return str.str();
}
/** Formatted output of the Window& type.
 *
 * @param[in] win Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const Window &win)
{
    std::stringstream str;
    str << win;
    return str.str();
}

/** Formatted output of the Window* type.
 *
 * @param[in] win Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(Window *win)
{
    std::string ret_str = "nullptr";
    if(win != nullptr)
    {
        std::stringstream str;
        str << *win;
        ret_str = str.str();
    }
    return ret_str;
}

/** Formatted output of the Rectangle type.
 *
 * @param[out] os   Output stream.
 * @param[in]  rect Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const Rectangle &rect)
{
    os << rect.width << "x" << rect.height;
    os << "+" << rect.x << "+" << rect.y;

    return os;
}

/** Formatted output of the PaddingMode type.
 *
 * @param[out] os   Output stream.
 * @param[in]  mode Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const PaddingMode &mode)
{
    switch(mode)
    {
        case PaddingMode::CONSTANT:
            os << "CONSTANT";
            break;
        case PaddingMode::REFLECT:
            os << "REFLECT";
            break;
        case PaddingMode::SYMMETRIC:
            os << "SYMMETRIC";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the PaddingMode type.
 *
 * @param[in] mode Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const PaddingMode &mode)
{
    std::stringstream str;
    str << mode;
    return str.str();
}

/** Formatted output of the PadStrideInfo type.
 *
 * @param[out] os              Output stream.
 * @param[in]  pad_stride_info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const PadStrideInfo &pad_stride_info)
{
    os << pad_stride_info.stride().first << "," << pad_stride_info.stride().second;
    os << ";";
    os << pad_stride_info.pad_left() << "," << pad_stride_info.pad_right() << ","
       << pad_stride_info.pad_top() << "," << pad_stride_info.pad_bottom();

    return os;
}

/** Formatted output of the PadStrideInfo type.
 *
 * @param[in] pad_stride_info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const PadStrideInfo &pad_stride_info)
{
    std::stringstream str;
    str << pad_stride_info;
    return str.str();
}

/** Formatted output of the BorderMode type.
 *
 * @param[in] mode Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const BorderMode &mode)
{
    std::stringstream str;
    str << mode;
    return str.str();
}

/** Formatted output of the BorderSize type.
 *
 * @param[in] border Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const BorderSize &border)
{
    std::stringstream str;
    str << border;
    return str.str();
}

/** Formatted output of the PaddingList type.
 *
 * @param[in] padding Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const PaddingList &padding)
{
    std::stringstream str;
    str << padding;
    return str.str();
}

/** Formatted output of the Multiples type.
 *
 * @param[in] multiples Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const Multiples &multiples)
{
    std::stringstream str;
    str << multiples;
    return str.str();
}

/** Formatted output of the InterpolationPolicy type.
 *
 * @param[in] policy Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const InterpolationPolicy &policy)
{
    std::stringstream str;
    str << policy;
    return str.str();
}

/** Formatted output of the SamplingPolicy type.
 *
 * @param[in] policy Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const SamplingPolicy &policy)
{
    std::stringstream str;
    str << policy;
    return str.str();
}

/** Formatted output of the ConvertPolicy type.
 *
 * @param[out] os     Output stream.
 * @param[in]  policy Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const ConvertPolicy &policy)
{
    switch(policy)
    {
        case ConvertPolicy::WRAP:
            os << "WRAP";
            break;
        case ConvertPolicy::SATURATE:
            os << "SATURATE";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

inline std::string to_string(const ConvertPolicy &policy)
{
    std::stringstream str;
    str << policy;
    return str.str();
}

/** Formatted output of the ArithmeticOperation type.
 *
 * @param[out] os Output stream.
 * @param[in]  op Operation to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const ArithmeticOperation &op)
{
    switch(op)
    {
        case ArithmeticOperation::ADD:
            os << "ADD";
            break;
        case ArithmeticOperation::SUB:
            os << "SUB";
            break;
        case ArithmeticOperation::DIV:
            os << "DIV";
            break;
        case ArithmeticOperation::MAX:
            os << "MAX";
            break;
        case ArithmeticOperation::MIN:
            os << "MIN";
            break;
        case ArithmeticOperation::SQUARED_DIFF:
            os << "SQUARED_DIFF";
            break;
        case ArithmeticOperation::POWER:
            os << "POWER";
            break;
        case ArithmeticOperation::PRELU:
            os << "PRELU";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the Arithmetic Operation
 *
 * @param[in] op Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const ArithmeticOperation &op)
{
    std::stringstream str;
    str << op;
    return str.str();
}

/** Formatted output of the Reduction Operations.
 *
 * @param[out] os Output stream.
 * @param[in]  op Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const ReductionOperation &op)
{
    switch(op)
    {
        case ReductionOperation::SUM:
            os << "SUM";
            break;
        case ReductionOperation::SUM_SQUARE:
            os << "SUM_SQUARE";
            break;
        case ReductionOperation::MEAN_SUM:
            os << "MEAN_SUM";
            break;
        case ReductionOperation::ARG_IDX_MAX:
            os << "ARG_IDX_MAX";
            break;
        case ReductionOperation::ARG_IDX_MIN:
            os << "ARG_IDX_MIN";
            break;
        case ReductionOperation::PROD:
            os << "PROD";
            break;
        case ReductionOperation::MIN:
            os << "MIN";
            break;
        case ReductionOperation::MAX:
            os << "MAX";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the Reduction Operations.
 *
 * @param[in] op Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const ReductionOperation &op)
{
    std::stringstream str;
    str << op;
    return str.str();
}

/** Formatted output of the Comparison Operations.
 *
 * @param[out] os Output stream.
 * @param[in]  op Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const ComparisonOperation &op)
{
    switch(op)
    {
        case ComparisonOperation::Equal:
            os << "Equal";
            break;
        case ComparisonOperation::NotEqual:
            os << "NotEqual";
            break;
        case ComparisonOperation::Greater:
            os << "Greater";
            break;
        case ComparisonOperation::GreaterEqual:
            os << "GreaterEqual";
            break;
        case ComparisonOperation::Less:
            os << "Less";
            break;
        case ComparisonOperation::LessEqual:
            os << "LessEqual";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the Elementwise unary Operations.
 *
 * @param[out] os Output stream.
 * @param[in]  op Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const ElementWiseUnary &op)
{
    switch(op)
    {
        case ElementWiseUnary::RSQRT:
            os << "RSQRT";
            break;
        case ElementWiseUnary::EXP:
            os << "EXP";
            break;
        case ElementWiseUnary::NEG:
            os << "NEG";
            break;
        case ElementWiseUnary::LOG:
            os << "LOG";
            break;
        case ElementWiseUnary::SIN:
            os << "SIN";
            break;
        case ElementWiseUnary::ABS:
            os << "ABS";
            break;
        case ElementWiseUnary::ROUND:
            os << "ROUND";
            break;
        case ElementWiseUnary::LOGICAL_NOT:
            os << "LOGICAL_NOT";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the Comparison Operations.
 *
 * @param[in] op Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const ComparisonOperation &op)
{
    std::stringstream str;
    str << op;
    return str.str();
}

/** Formatted output of the Elementwise unary Operations.
 *
 * @param[in] op Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const ElementWiseUnary &op)
{
    std::stringstream str;
    str << op;
    return str.str();
}

/** Formatted output of the Norm Type.
 *
 * @param[in] type Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const NormType &type)
{
    std::stringstream str;
    str << type;
    return str.str();
}

/** Formatted output of the Pooling Type.
 *
 * @param[in] type Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const PoolingType &type)
{
    std::stringstream str;
    str << type;
    return str.str();
}

/** Formatted output of the Pooling Layer Info.
 *
 * @param[in] info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const PoolingLayerInfo &info)
{
    std::stringstream str;
    str << "{Type=" << info.pool_type << ","
        << "DataLayout=" << info.data_layout << ","
        << "IsGlobalPooling=" << info.is_global_pooling;
    if(!info.is_global_pooling)
    {
        str << ","
            << "PoolSize=" << info.pool_size.width << "," << info.pool_size.height << ","
            << "PadStride=" << info.pad_stride_info;
    }
    str << "}";
    return str.str();
}

/** Formatted output of the Size3D type.
 *
 * @param[out] os   Output stream
 * @param[in]  size Type to output
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const Size3D &size)
{
    os << size.width << "x" << size.height << "x" << size.depth;

    return os;
}

/** Formatted output of the Size3D type.
 *
 * @param[in] type Type to output
 *
 * @return Formatted string.
 */
inline std::string to_string(const Size3D &type)
{
    std::stringstream str;
    str << type;
    return str.str();
}

/** Formatted output of the Padding3D type.
 *
 * @param[out] os        Output stream.
 * @param[in]  padding3d Padding info for 3D spatial dimension shape.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const Padding3D &padding3d)
{
    os << padding3d.left << "," << padding3d.right << ","
       << padding3d.top << "," << padding3d.bottom << ","
       << padding3d.front << "," << padding3d.back;
    return os;
}

/** Converts a @ref Padding3D to string
 *
 * @param[in] padding3d Padding3D value to be converted
 *
 * @return String representing the corresponding Padding3D
 */
inline std::string to_string(const Padding3D &padding3d)
{
    std::stringstream str;
    str << padding3d;
    return str.str();
}

/** Formatted output of the DimensionRoundingType type.
 *
 * @param[out] os            Output stream.
 * @param[in]  rounding_type DimensionRoundingType Dimension rounding type when down-scaling, or compute output shape of pooling(2D or 3D).
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const DimensionRoundingType &rounding_type)
{
    switch(rounding_type)
    {
        case DimensionRoundingType::CEIL:
            os << "CEIL";
            break;
        case DimensionRoundingType::FLOOR:
            os << "FLOOR";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }
    return os;
}

/** Formatted output of the Pooling 3d Layer Info.
 *
 * @param[out] os   Output stream.
 * @param[in]  info Pooling 3D layer info to print to output stream.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const Pooling3dLayerInfo &info)
{
    os << "{Type=" << info.pool_type << ","
       << "IsGlobalPooling=" << info.is_global_pooling;
    if(!info.is_global_pooling)
    {
        os << ","
           << "PoolSize=" << info.pool_size << ", "
           << "Stride=" << info.stride << ", "
           << "Padding=" << info.padding << ", "
           << "Exclude Padding=" << info.exclude_padding << ", "
           << "fp_mixed_precision=" << info.fp_mixed_precision << ", "
           << "DimensionRoundingType=" << info.round_type;
    }
    os << "}";
    return os;
}

/** Formatted output of the Pooling 3d Layer Info.
 *
 * @param[in] info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const Pooling3dLayerInfo &info)
{
    std::stringstream str;
    str << info;
    return str.str();
}

/** Formatted output of the PriorBoxLayerInfo.
 *
 * @param[in] info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const PriorBoxLayerInfo &info)
{
    std::stringstream str;
    str << "{";
    str << "Clip:" << info.clip()
        << "Flip:" << info.flip()
        << "StepX:" << info.steps()[0]
        << "StepY:" << info.steps()[1]
        << "MinSizes:" << info.min_sizes().size()
        << "MaxSizes:" << info.max_sizes().size()
        << "ImgSizeX:" << info.img_size().x
        << "ImgSizeY:" << info.img_size().y
        << "Offset:" << info.offset()
        << "Variances:" << info.variances().size();
    str << "}";
    return str.str();
}

/** Formatted output of the Size2D type.
 *
 * @param[out] os   Output stream
 * @param[in]  size Type to output
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const Size2D &size)
{
    os << size.width << "x" << size.height;

    return os;
}

/** Formatted output of the Size2D type.
 *
 * @param[in] type Type to output
 *
 * @return Formatted string.
 */
inline std::string to_string(const Size2D &type)
{
    std::stringstream str;
    str << type;
    return str.str();
}

/** Formatted output of the ConvolutionMethod type.
 *
 * @param[out] os          Output stream
 * @param[in]  conv_method Type to output
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const ConvolutionMethod &conv_method)
{
    switch(conv_method)
    {
        case ConvolutionMethod::GEMM:
            os << "GEMM";
            break;
        case ConvolutionMethod::DIRECT:
            os << "DIRECT";
            break;
        case ConvolutionMethod::WINOGRAD:
            os << "WINOGRAD";
            break;
        case ConvolutionMethod::FFT:
            os << "FFT";
            break;
        case ConvolutionMethod::GEMM_CONV2D:
            os << "GEMM_CONV2D";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the ConvolutionMethod type.
 *
 * @param[in] conv_method Type to output
 *
 * @return Formatted string.
 */
inline std::string to_string(const ConvolutionMethod &conv_method)
{
    std::stringstream str;
    str << conv_method;
    return str.str();
}

/** Formatted output of the GPUTarget type.
 *
 * @param[out] os         Output stream
 * @param[in]  gpu_target Type to output
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const GPUTarget &gpu_target)
{
    switch(gpu_target)
    {
        case GPUTarget::GPU_ARCH_MASK:
            os << "GPU_ARCH_MASK";
            break;
        case GPUTarget::GPU_GENERATION_MASK:
            os << "GPU_GENERATION_MASK";
            break;
        case GPUTarget::MIDGARD:
            os << "MIDGARD";
            break;
        case GPUTarget::BIFROST:
            os << "BIFROST";
            break;
        case GPUTarget::VALHALL:
            os << "VALHALL";
            break;
        case GPUTarget::T600:
            os << "T600";
            break;
        case GPUTarget::T700:
            os << "T700";
            break;
        case GPUTarget::T800:
            os << "T800";
            break;
        case GPUTarget::G71:
            os << "G71";
            break;
        case GPUTarget::G72:
            os << "G72";
            break;
        case GPUTarget::G51:
            os << "G51";
            break;
        case GPUTarget::G51BIG:
            os << "G51BIG";
            break;
        case GPUTarget::G51LIT:
            os << "G51LIT";
            break;
        case GPUTarget::G31:
            os << "G31";
            break;
        case GPUTarget::G76:
            os << "G76";
            break;
        case GPUTarget::G52:
            os << "G52";
            break;
        case GPUTarget::G52LIT:
            os << "G52LIT";
            break;
        case GPUTarget::G77:
            os << "G77";
            break;
        case GPUTarget::G57:
            os << "G57";
            break;
        case GPUTarget::G78:
            os << "G78";
            break;
        case GPUTarget::G68:
            os << "G68";
            break;
        case GPUTarget::G78AE:
            os << "G78AE";
            break;
        case GPUTarget::G710:
            os << "G710";
            break;
        case GPUTarget::G610:
            os << "G610";
            break;
        case GPUTarget::G510:
            os << "G510";
            break;
        case GPUTarget::G310:
            os << "G310";
            break;
        case GPUTarget::G715:
            os << "G715";
            break;
        case GPUTarget::G615:
            os << "G615";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the GPUTarget type.
 *
 * @param[in] gpu_target Type to output
 *
 * @return Formatted string.
 */
inline std::string to_string(const GPUTarget &gpu_target)
{
    std::stringstream str;
    str << gpu_target;
    return str.str();
}

/** Formatted output of the DetectionWindow type.
 *
 * @param[out] os               Output stream
 * @param[in]  detection_window Type to output
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const DetectionWindow &detection_window)
{
    os << "{x=" << detection_window.x << ","
       << "y=" << detection_window.y << ","
       << "width=" << detection_window.width << ","
       << "height=" << detection_window.height << ","
       << "idx_class=" << detection_window.idx_class << ","
       << "score=" << detection_window.score << "}";

    return os;
}

/** Formatted output of the DetectionOutputLayerCodeType type.
 *
 * @param[out] os             Output stream
 * @param[in]  detection_code Type to output
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const DetectionOutputLayerCodeType &detection_code)
{
    switch(detection_code)
    {
        case DetectionOutputLayerCodeType::CENTER_SIZE:
            os << "CENTER_SIZE";
            break;
        case DetectionOutputLayerCodeType::CORNER:
            os << "CORNER";
            break;
        case DetectionOutputLayerCodeType::CORNER_SIZE:
            os << "CORNER_SIZE";
            break;
        case DetectionOutputLayerCodeType::TF_CENTER:
            os << "TF_CENTER";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}
/** Formatted output of the DetectionOutputLayerCodeType type.
 *
 * @param[in] detection_code Type to output
 *
 * @return Formatted string.
 */
inline std::string to_string(const DetectionOutputLayerCodeType &detection_code)
{
    std::stringstream str;
    str << detection_code;
    return str.str();
}

/** Formatted output of the DetectionOutputLayerInfo type.
 *
 * @param[out] os             Output stream
 * @param[in]  detection_info Type to output
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const DetectionOutputLayerInfo &detection_info)
{
    os << "{Classes=" << detection_info.num_classes() << ","
       << "ShareLocation=" << detection_info.share_location() << ","
       << "CodeType=" << detection_info.code_type() << ","
       << "VarianceEncodedInTarget=" << detection_info.variance_encoded_in_target() << ","
       << "KeepTopK=" << detection_info.keep_top_k() << ","
       << "NMSThreshold=" << detection_info.nms_threshold() << ","
       << "Eta=" << detection_info.eta() << ","
       << "BackgroundLabelId=" << detection_info.background_label_id() << ","
       << "ConfidenceThreshold=" << detection_info.confidence_threshold() << ","
       << "TopK=" << detection_info.top_k() << ","
       << "NumLocClasses=" << detection_info.num_loc_classes()
       << "}";

    return os;
}

/** Formatted output of the DetectionOutputLayerInfo type.
 *
 * @param[in] detection_info Type to output
 *
 * @return Formatted string.
 */
inline std::string to_string(const DetectionOutputLayerInfo &detection_info)
{
    std::stringstream str;
    str << detection_info;
    return str.str();
}
/** Formatted output of the DetectionPostProcessLayerInfo type.
 *
 * @param[out] os             Output stream
 * @param[in]  detection_info Type to output
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const DetectionPostProcessLayerInfo &detection_info)
{
    os << "{MaxDetections=" << detection_info.max_detections() << ","
       << "MaxClassesPerDetection=" << detection_info.max_classes_per_detection() << ","
       << "NmsScoreThreshold=" << detection_info.nms_score_threshold() << ","
       << "NmsIouThreshold=" << detection_info.iou_threshold() << ","
       << "NumClasses=" << detection_info.num_classes() << ","
       << "ScaleValue_y=" << detection_info.scale_value_y() << ","
       << "ScaleValue_x=" << detection_info.scale_value_x() << ","
       << "ScaleValue_h=" << detection_info.scale_value_h() << ","
       << "ScaleValue_w=" << detection_info.scale_value_w() << ","
       << "UseRegularNms=" << detection_info.use_regular_nms() << ","
       << "DetectionPerClass=" << detection_info.detection_per_class()
       << "}";

    return os;
}

/** Formatted output of the DetectionPostProcessLayerInfo type.
 *
 * @param[in] detection_info Type to output
 *
 * @return Formatted string.
 */
inline std::string to_string(const DetectionPostProcessLayerInfo &detection_info)
{
    std::stringstream str;
    str << detection_info;
    return str.str();
}

/** Formatted output of the DetectionWindow type.
 *
 * @param[in] detection_window Type to output
 *
 * @return Formatted string.
 */
inline std::string to_string(const DetectionWindow &detection_window)
{
    std::stringstream str;
    str << detection_window;
    return str.str();
}

/** Formatted output of @ref PriorBoxLayerInfo.
 *
 * @param[out] os   Output stream.
 * @param[in]  info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const PriorBoxLayerInfo &info)
{
    os << "Clip:" << info.clip()
       << "Flip:" << info.flip()
       << "StepX:" << info.steps()[0]
       << "StepY:" << info.steps()[1]
       << "MinSizes:" << info.min_sizes()
       << "MaxSizes:" << info.max_sizes()
       << "ImgSizeX:" << info.img_size().x
       << "ImgSizeY:" << info.img_size().y
       << "Offset:" << info.offset()
       << "Variances:" << info.variances();

    return os;
}

/** Formatted output of the WinogradInfo type. */
inline ::std::ostream &operator<<(::std::ostream &os, const WinogradInfo &info)
{
    os << "{OutputTileSize=" << info.output_tile_size << ","
       << "KernelSize=" << info.kernel_size << ","
       << "PadStride=" << info.convolution_info << ","
       << "OutputDataLayout=" << info.output_data_layout << "}";

    return os;
}

inline std::string to_string(const WinogradInfo &type)
{
    std::stringstream str;
    str << type;
    return str.str();
}

/** Convert a CLTunerMode value to a string
 *
 * @param val CLTunerMode value to be converted
 *
 * @return String representing the corresponding CLTunerMode.
 */
inline std::string to_string(const CLTunerMode val)
{
    switch(val)
    {
        case CLTunerMode::EXHAUSTIVE:
        {
            return std::string("Exhaustive");
        }
        case CLTunerMode::NORMAL:
        {
            return std::string("Normal");
        }
        case CLTunerMode::RAPID:
        {
            return std::string("Rapid");
        }
        default:
        {
            ARM_COMPUTE_ERROR("Invalid tuner mode.");
            return std::string("UNDEFINED");
        }
    }
}
/** Converts a @ref CLGEMMKernelType to string
 *
 * @param[in] val CLGEMMKernelType value to be converted
 *
 * @return String representing the corresponding CLGEMMKernelType
 */
inline std::string to_string(CLGEMMKernelType val)
{
    switch(val)
    {
        case CLGEMMKernelType::NATIVE:
        {
            return "Native";
        }
        case CLGEMMKernelType::RESHAPED_ONLY_RHS:
        {
            return "Reshaped_Only_RHS";
        }
        case CLGEMMKernelType::RESHAPED:
        {
            return "Reshaped";
        }
        default:
        {
            return "Unknown";
        }
    }
}
/** [Print CLTunerMode type] **/
/** Formatted output of the CLTunerMode type.
 *
 * @param[out] os  Output stream.
 * @param[in]  val CLTunerMode to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const CLTunerMode &val)
{
    os << to_string(val);
    return os;
}

/** Formatted output of the ConvolutionInfo type.
 *
 * @param[out] os        Output stream.
 * @param[in]  conv_info ConvolutionInfo to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const ConvolutionInfo &conv_info)
{
    os << "{PadStrideInfo=" << conv_info.pad_stride_info << ", "
       << "depth_multiplier=" << conv_info.depth_multiplier << ", "
       << "act_info=" << to_string(conv_info.act_info) << ", "
       << "dilation=" << conv_info.dilation << "}";
    return os;
}

/** Converts a @ref ConvolutionInfo to string
 *
 * @param[in] info ConvolutionInfo value to be converted
 *
 * @return String  representing the corresponding ConvolutionInfo
 */
inline std::string to_string(const ConvolutionInfo &info)
{
    std::stringstream str;
    str << info;
    return str.str();
}

/** Formatted output of the FullyConnectedLayerInfo type.
 *
 * @param[out] os         Output stream.
 * @param[in]  layer_info FullyConnectedLayerInfo to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const FullyConnectedLayerInfo &layer_info)
{
    os << "{activation_info=" << to_string(layer_info.activation_info) << ", "
       << "weights_trained_layout=" << layer_info.weights_trained_layout << ", "
       << "transpose_weights=" << layer_info.transpose_weights << ", "
       << "are_weights_reshaped=" << layer_info.are_weights_reshaped << ", "
       << "retain_internal_weights=" << layer_info.retain_internal_weights << ", "
       << "fp_mixed_precision=" << layer_info.fp_mixed_precision << "}";
    return os;
}

/** Converts a @ref FullyConnectedLayerInfo to string
 *
 * @param[in] info FullyConnectedLayerInfo value to be converted
 *
 * @return String  representing the corresponding FullyConnectedLayerInfo
 */
inline std::string to_string(const FullyConnectedLayerInfo &info)
{
    std::stringstream str;
    str << info;
    return str.str();
}

/** Formatted output of the GEMMLowpOutputStageType type.
 *
 * @param[out] os        Output stream.
 * @param[in]  gemm_type GEMMLowpOutputStageType to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const GEMMLowpOutputStageType &gemm_type)
{
    switch(gemm_type)
    {
        case GEMMLowpOutputStageType::NONE:
            os << "NONE";
            break;
        case GEMMLowpOutputStageType::QUANTIZE_DOWN:
            os << "QUANTIZE_DOWN";
            break;
        case GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT:
            os << "QUANTIZE_DOWN_FIXEDPOINT";
            break;
        case GEMMLowpOutputStageType::QUANTIZE_DOWN_FLOAT:
            os << "QUANTIZE_DOWN_FLOAT";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }
    return os;
}

/** Converts a @ref GEMMLowpOutputStageType to string
 *
 * @param[in] gemm_type GEMMLowpOutputStageType value to be converted
 *
 * @return String       representing the corresponding GEMMLowpOutputStageType
 */
inline std::string to_string(const GEMMLowpOutputStageType &gemm_type)
{
    std::stringstream str;
    str << gemm_type;
    return str.str();
}

/** Formatted output of the GEMMLowpOutputStageInfo type.
 *
 * @param[out] os        Output stream.
 * @param[in]  gemm_info GEMMLowpOutputStageInfo to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const GEMMLowpOutputStageInfo &gemm_info)
{
    os << "{type=" << gemm_info.type << ", "
       << "gemlowp_offset=" << gemm_info.gemmlowp_offset << ", "
       << "gemmlowp_multiplier=" << gemm_info.gemmlowp_multiplier << ", "
       << "gemmlowp_shift=" << gemm_info.gemmlowp_shift << ", "
       << "gemmlowp_min_bound=" << gemm_info.gemmlowp_min_bound << ", "
       << "gemmlowp_max_bound=" << gemm_info.gemmlowp_max_bound << ", "
       << "gemmlowp_multipliers=" << gemm_info.gemmlowp_multiplier << ", "
       << "gemmlowp_shifts=" << gemm_info.gemmlowp_shift << ", "
       << "gemmlowp_real_multiplier=" << gemm_info.gemmlowp_real_multiplier << ", "
       << "is_quantized_per_channel=" << gemm_info.is_quantized_per_channel << ", "
       << "output_data_type=" << gemm_info.output_data_type << "}";
    return os;
}

/** Converts a @ref GEMMLowpOutputStageInfo to string
 *
 * @param[in] gemm_info GEMMLowpOutputStageInfo value to be converted
 *
 * @return String representing the corresponding GEMMLowpOutputStageInfo
 */
inline std::string to_string(const GEMMLowpOutputStageInfo &gemm_info)
{
    std::stringstream str;
    str << gemm_info;
    return str.str();
}

/** Formatted output of the Conv2dInfo type.
 *
 * @param[out] os        Output stream.
 * @param[in]  conv_info Conv2dInfo to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const Conv2dInfo &conv_info)
{
    os << "{conv_info=" << conv_info.conv_info << ", "
       << "dilation=" << conv_info.dilation << ", "
       << "act_info=" << to_string(conv_info.act_info) << ", "
       << "enable_fast_math=" << conv_info.enable_fast_math << ", "
       << "num_groups=" << conv_info.num_groups << ","
       << "post_ops=" << conv_info.post_ops << "}";
    return os;
}

/** Converts a @ref Conv2dInfo to string
 *
 * @param[in] conv_info Conv2dInfo value to be converted
 *
 * @return String  representing the corresponding Conv2dInfo
 */
inline std::string to_string(const Conv2dInfo &conv_info)
{
    std::stringstream str;
    str << conv_info;
    return str.str();
}

/** Formatted output of the PixelValue type.
 *
 * @param[out] os          Output stream.
 * @param[in]  pixel_value PixelValue to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const PixelValue &pixel_value)
{
    os << "{value.u64=" << pixel_value.get<uint64_t>() << "}";
    return os;
}

/** Converts a @ref PixelValue to string
 *
 * @param[in] pixel_value PixelValue value to be converted
 *
 * @return String representing the corresponding PixelValue
 */
inline std::string to_string(const PixelValue &pixel_value)
{
    std::stringstream str;
    str << pixel_value;
    return str.str();
}

/** Formatted output of the ScaleKernelInfo type.
 *
 * @param[out] os         Output stream.
 * @param[in]  scale_info ScaleKernelInfo to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const ScaleKernelInfo &scale_info)
{
    os << "{interpolation_policy=" << scale_info.interpolation_policy << ", "
       << "BorderMode=" << scale_info.border_mode << ", "
       << "PixelValue=" << scale_info.constant_border_value << ", "
       << "SamplingPolicy=" << scale_info.sampling_policy << ", "
       << "use_padding=" << scale_info.use_padding << ", "
       << "align_corners=" << scale_info.align_corners << ", "
       << "data_layout=" << scale_info.data_layout << "}";
    return os;
}

/** Converts a @ref ScaleKernelInfo to string
 *
 * @param[in] scale_info ScaleKernelInfo value to be converted
 *
 * @return String representing the corresponding ScaleKernelInfo
 */
inline std::string to_string(const ScaleKernelInfo &scale_info)
{
    std::stringstream str;
    str << scale_info;
    return str.str();
}

/** Formatted output of the FFTDirection type.
 *
 * @param[out] os      Output stream.
 * @param[in]  fft_dir FFTDirection to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const FFTDirection &fft_dir)
{
    switch(fft_dir)
    {
        case FFTDirection::Forward:
            os << "Forward";
            break;
        case FFTDirection::Inverse:
            os << "Inverse";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }
    return os;
}

/** Converts a @ref FFT1DInfo to string
 *
 * @param[in] fft_dir FFT1DInfo value to be converted
 *
 * @return String representing the corresponding FFT1DInfo
 */
inline std::string to_string(const FFTDirection &fft_dir)
{
    std::stringstream str;
    str << "{" << fft_dir << "}";
    return str.str();
}

/** Formatted output of the FFT1DInfo type.
 *
 * @param[out] os         Output stream.
 * @param[in]  fft1d_info FFT1DInfo to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const FFT1DInfo &fft1d_info)
{
    os << "{axis=" << fft1d_info.axis << ", "
       << "direction=" << fft1d_info.direction << "}";
    return os;
}

/** Converts a @ref FFT1DInfo to string
 *
 * @param[in] fft1d_info FFT1DInfo value to be converted
 *
 * @return String representing the corresponding FFT1DInfo
 */
inline std::string to_string(const FFT1DInfo &fft1d_info)
{
    std::stringstream str;
    str << fft1d_info;
    return str.str();
}

/** Formatted output of the FFT2DInfo type.
 *
 * @param[out] os         Output stream.
 * @param[in]  fft2d_info FFT2DInfo to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const FFT2DInfo &fft2d_info)
{
    os << "{axis=" << fft2d_info.axis0 << ", "
       << "axis=" << fft2d_info.axis1 << ", "
       << "direction=" << fft2d_info.direction << "}";
    return os;
}

/** Converts a @ref FFT2DInfo to string
 *
 * @param[in] fft2d_info FFT2DInfo value to be converted
 *
 * @return String representing the corresponding FFT2DInfo
 */
inline std::string to_string(const FFT2DInfo &fft2d_info)
{
    std::stringstream str;
    str << fft2d_info;
    return str.str();
}

/** Formatted output of the Coordinates2D type.
 *
 * @param[out] os       Output stream.
 * @param[in]  coord_2d Coordinates2D to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const Coordinates2D &coord_2d)
{
    os << "{x=" << coord_2d.x << ", "
       << "y=" << coord_2d.y << "}";
    return os;
}

/** Converts a @ref Coordinates2D to string
 *
 * @param[in] coord_2d Coordinates2D value to be converted
 *
 * @return String representing the corresponding Coordinates2D
 */
inline std::string to_string(const Coordinates2D &coord_2d)
{
    std::stringstream str;
    str << coord_2d;
    return str.str();
}

/** Formatted output of the FuseBatchNormalizationType type.
 *
 * @param[out] os        Output stream.
 * @param[in]  fuse_type FuseBatchNormalizationType to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const FuseBatchNormalizationType &fuse_type)
{
    switch(fuse_type)
    {
        case FuseBatchNormalizationType::CONVOLUTION:
            os << "CONVOLUTION";
            break;
        case FuseBatchNormalizationType::DEPTHWISECONVOLUTION:
            os << "DEPTHWISECONVOLUTION";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }
    return os;
}

/** Converts a @ref FuseBatchNormalizationType to string
 *
 * @param[in] fuse_type FuseBatchNormalizationType value to be converted
 *
 * @return String representing the corresponding FuseBatchNormalizationType
 */
inline std::string to_string(const FuseBatchNormalizationType &fuse_type)
{
    std::stringstream str;
    str << fuse_type;
    return str.str();
}

/** Formatted output of the SoftmaxKernelInfo type.
 *
 * @param[out] os   Output stream.
 * @param[in]  info SoftmaxKernelInfo to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const SoftmaxKernelInfo &info)
{
    os << "{beta=" << info.beta << ", "
       << "is_log=" << info.is_log << ", "
       << "input_data_type=" << info.input_data_type << ", "
       << "axis=" << info.axis << "}";
    return os;
}

/** Converts a @ref SoftmaxKernelInfo to string
 *
 * @param[in] info SoftmaxKernelInfo value to be converted
 *
 * @return String representing the corresponding SoftmaxKernelInfo
 */
inline std::string to_string(const SoftmaxKernelInfo &info)
{
    std::stringstream str;
    str << info;
    return str.str();
}

/** Formatted output of the ScaleKernelInfo type.
 *
 * @param[out] os          Output stream.
 * @param[in]  lstm_params LSTMParams to output.
 *
 * @return Modified output stream.
 */
template <typename T>
::std::ostream &operator<<(::std::ostream &os, const LSTMParams<T> &lstm_params)
{
    os << "{input_to_input_weights=" << to_string(lstm_params.input_to_input_weights()) << ", "
       << "recurrent_to_input_weights=" << to_string(lstm_params.recurrent_to_input_weights()) << ", "
       << "cell_to_input_weights=" << to_string(lstm_params.cell_to_input_weights()) << ", "
       << "input_gate_bias=" << to_string(lstm_params.input_gate_bias()) << ", "
       << "cell_to_forget_weights=" << to_string(lstm_params.cell_to_forget_weights()) << ", "
       << "cell_to_output_weights=" << to_string(lstm_params.cell_to_output_weights()) << ", "
       << "projection_weights=" << to_string(lstm_params.projection_weights()) << ", "
       << "projection_bias=" << to_string(lstm_params.projection_bias()) << ", "
       << "input_layer_norm_weights=" << to_string(lstm_params.input_layer_norm_weights()) << ", "
       << "forget_layer_norm_weights=" << to_string(lstm_params.forget_layer_norm_weights()) << ", "
       << "cell_layer_norm_weights=" << to_string(lstm_params.cell_layer_norm_weights()) << ", "
       << "output_layer_norm_weights=" << to_string(lstm_params.output_layer_norm_weights()) << ", "
       << "cell_clip=" << lstm_params.cell_clip() << ", "
       << "projection_clip=" << lstm_params.projection_clip() << ", "
       << "input_intermediate_scale=" << lstm_params.input_intermediate_scale() << ", "
       << "forget_intermediate_scale=" << lstm_params.forget_intermediate_scale() << ", "
       << "cell_intermediate_scale=" << lstm_params.cell_intermediate_scale() << ", "
       << "hidden_state_zero=" << lstm_params.hidden_state_zero() << ", "
       << "hidden_state_scale=" << lstm_params.hidden_state_scale() << ", "
       << "has_peephole_opt=" << lstm_params.has_peephole_opt() << ", "
       << "has_projection=" << lstm_params.has_projection() << ", "
       << "has_cifg_opt=" << lstm_params.has_cifg_opt() << ", "
       << "use_layer_norm=" << lstm_params.use_layer_norm() << "}";
    return os;
}

/** Converts a @ref LSTMParams to string
 *
 * @param[in] lstm_params LSTMParams<T> value to be converted
 *
 * @return String representing the corresponding LSTMParams
 */
template <typename T>
std::string to_string(const LSTMParams<T> &lstm_params)
{
    std::stringstream str;
    str << lstm_params;
    return str.str();
}

/** Converts a @ref LSTMParams to string
 *
 * @param[in] num uint8_t value to be converted
 *
 * @return String representing the corresponding uint8_t
 */
inline std::string to_string(const uint8_t num)
{
    // Explicity cast the uint8_t to signed integer and call the corresponding overloaded to_string() function.
    return ::std::to_string(static_cast<int>(num));
}

/** Available non maxima suppression types */
/** Formatted output of the NMSType type.
 *
 * @param[out] os       Output stream.
 * @param[in]  nms_type NMSType to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const NMSType &nms_type)
{
    switch(nms_type)
    {
        case NMSType::LINEAR:
            os << "LINEAR";
            break;
        case NMSType::GAUSSIAN:
            os << "GAUSSIAN";
            break;
        case NMSType::ORIGINAL:
            os << "ORIGINAL";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }
    return os;
}

/** Converts a @ref NMSType to string
 *
 * @param[in] nms_type NMSType value to be converted
 *
 * @return String representing the corresponding NMSType
 */
inline std::string to_string(const NMSType nms_type)
{
    std::stringstream str;
    str << nms_type;
    return str.str();
}

/** Formatted output of the BoxNMSLimitInfo type.
 *
 * @param[out] os   Output stream.
 * @param[in]  info BoxNMSLimitInfo to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const BoxNMSLimitInfo &info)
{
    os << "{score_thresh=" << info.score_thresh() << ", "
       << "nms=" << info.nms() << ", "
       << "detections_per_im=" << info.detections_per_im() << ", "
       << "soft_nms_enabled=" << info.soft_nms_enabled() << ", "
       << "soft_nms_min_score_thres=" << info.soft_nms_min_score_thres() << ", "
       << "suppress_size=" << info.suppress_size() << ", "
       << "min_size=" << info.min_size() << ", "
       << "im_width=" << info.im_width() << ", "
       << "im_height=" << info.im_height() << "}";
    return os;
}

/** Converts a @ref BoxNMSLimitInfo to string
 *
 * @param[in] info BoxNMSLimitInfo value to be converted
 *
 * @return String representing the corresponding BoxNMSLimitInfo
 */
inline std::string to_string(const BoxNMSLimitInfo &info)
{
    std::stringstream str;
    str << info;
    return str.str();
}

/** Converts a @ref DimensionRoundingType to string
 *
 * @param[in] rounding_type DimensionRoundingType value to be converted
 *
 * @return String representing the corresponding DimensionRoundingType
 */
inline std::string to_string(const DimensionRoundingType &rounding_type)
{
    std::stringstream str;
    str << rounding_type;
    return str.str();
}

/** Formatted output of the Conv3dInfo type.
 *
 * @param[out] os          Output stream.
 * @param[in]  conv3d_info Type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const Conv3dInfo &conv3d_info)
{
    os << conv3d_info.stride;
    os << ";";
    os << conv3d_info.padding;
    os << ";";
    os << to_string(conv3d_info.act_info);
    os << ";";
    os << conv3d_info.dilation;
    os << ";";
    os << conv3d_info.round_type;
    os << ";";
    os << conv3d_info.enable_fast_math;

    return os;
}

/** Formatted output of the Conv3dInfo type.
 *
 * @param[in] conv3d_info Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const Conv3dInfo &conv3d_info)
{
    std::stringstream str;
    str << conv3d_info;
    return str.str();
}

/** Formatted output of the arm_compute::WeightFormat type.
 *
 * @param[in] wf arm_compute::WeightFormat Type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const WeightFormat wf)
{
#define __CASE_WEIGHT_FORMAT(wf) \
case WeightFormat::wf:       \
    return #wf;
    switch(wf)
    {
            __CASE_WEIGHT_FORMAT(UNSPECIFIED)
            __CASE_WEIGHT_FORMAT(ANY)
            __CASE_WEIGHT_FORMAT(OHWI)
            __CASE_WEIGHT_FORMAT(OHWIo2)
            __CASE_WEIGHT_FORMAT(OHWIo4)
            __CASE_WEIGHT_FORMAT(OHWIo8)
            __CASE_WEIGHT_FORMAT(OHWIo16)
            __CASE_WEIGHT_FORMAT(OHWIo32)
            __CASE_WEIGHT_FORMAT(OHWIo64)
            __CASE_WEIGHT_FORMAT(OHWIo128)
            __CASE_WEIGHT_FORMAT(OHWIo4i2)
            __CASE_WEIGHT_FORMAT(OHWIo4i2_bf16)
            __CASE_WEIGHT_FORMAT(OHWIo8i2)
            __CASE_WEIGHT_FORMAT(OHWIo8i2_bf16)
            __CASE_WEIGHT_FORMAT(OHWIo16i2)
            __CASE_WEIGHT_FORMAT(OHWIo16i2_bf16)
            __CASE_WEIGHT_FORMAT(OHWIo32i2)
            __CASE_WEIGHT_FORMAT(OHWIo32i2_bf16)
            __CASE_WEIGHT_FORMAT(OHWIo64i2)
            __CASE_WEIGHT_FORMAT(OHWIo64i2_bf16)
            __CASE_WEIGHT_FORMAT(OHWIo4i4)
            __CASE_WEIGHT_FORMAT(OHWIo4i4_bf16)
            __CASE_WEIGHT_FORMAT(OHWIo8i4)
            __CASE_WEIGHT_FORMAT(OHWIo8i4_bf16)
            __CASE_WEIGHT_FORMAT(OHWIo16i4)
            __CASE_WEIGHT_FORMAT(OHWIo16i4_bf16)
            __CASE_WEIGHT_FORMAT(OHWIo32i4)
            __CASE_WEIGHT_FORMAT(OHWIo32i4_bf16)
            __CASE_WEIGHT_FORMAT(OHWIo64i4)
            __CASE_WEIGHT_FORMAT(OHWIo64i4_bf16)
            __CASE_WEIGHT_FORMAT(OHWIo2i8)
            __CASE_WEIGHT_FORMAT(OHWIo4i8)
            __CASE_WEIGHT_FORMAT(OHWIo8i8)
            __CASE_WEIGHT_FORMAT(OHWIo16i8)
            __CASE_WEIGHT_FORMAT(OHWIo32i8)
            __CASE_WEIGHT_FORMAT(OHWIo64i8)
        default:
            return "invalid value";
    }
#undef __CASE_WEIGHT_FORMAT
}

/** Formatted output of the arm_compute::WeightFormat type.
 *
 * @param[out] os Output stream.
 * @param[in]  wf WeightFormat to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const arm_compute::WeightFormat &wf)
{
    os << to_string(wf);
    return os;
}

/** Formatted output of the std::tuple<TensorShape, TensorShape, arm_compute::WeightFormat> tuple.
 *
 * @param[in] values tuple of input and output tensor shapes and WeightFormat used.
 *
 * @return Formatted string.
 */
inline std::string to_string(const std::tuple<TensorShape, TensorShape, arm_compute::WeightFormat> values)
{
    std::stringstream str;
    str << "[Input shape = " << std::get<0>(values);
    str << ", ";
    str << "Expected output shape = " << std::get<1>(values);

    str << ", ";
    str << "WeightFormat = " << std::get<2>(values) << "]";
    return str.str();
}

/** Formatted output of the Padding2D type.
 *
 * @param[out] os        Output stream.
 * @param[in]  padding2d Padding info for 2D dimension shape.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const Padding2D &padding2d)
{
    os << padding2d.left << "," << padding2d.right << ","
       << padding2d.top << "," << padding2d.bottom;
    return os;
}

/** Converts a @ref Padding2D to string
 *
 * @param[in] padding2d Padding2D value to be converted
 *
 * @return String representing the corresponding Padding2D
 */
inline std::string to_string(const Padding2D &padding2d)
{
    std::stringstream str;
    str << padding2d;
    return str.str();
}

/** Formatted output of the arm_compute::experimental::dynamic_fusion::Pool2dAttributes type.
 *
 * @param[out] os          Output stream.
 * @param[in]  pool2d_attr arm_compute::experimental::dynamic_fusion::Pool2dAttributes type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const experimental::dynamic_fusion::Pool2dAttributes &pool2d_attr)
{
    os << "Pool2dAttributes="
       << "["
       << "PoolingType=" << pool2d_attr.pool_type() << ","
       << "PoolSize=" << pool2d_attr.pool_size() << ","
       << "Padding=" << pool2d_attr.pad() << ","
       << "Stride=" << pool2d_attr.stride() << ","
       << "ExcludePadding" << pool2d_attr.exclude_padding() << "]";

    return os;
}

/** Formatted output of the arm_compute::experimental::dynamic_fusion::Pool2dAttributes type.
 *
 * @param[in] pool2d_attr arm_compute::experimental::dynamic_fusion::Pool2dAttributes type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const experimental::dynamic_fusion::Pool2dAttributes &pool2d_attr)
{
    std::stringstream str;
    str << pool2d_attr;
    return str.str();
}

/** Formatted output of the arm_compute::experimental::dynamic_fusion::GpuPool2dSettings type
 *
 * @param[out] os       Output stream
 * @param[in]  settings arm_compute::dynamic_fusion::GpuPool2dSettings type to output
 */
inline ::std::ostream &operator<<(::std::ostream &os, const experimental::dynamic_fusion::GpuPool2dSettings &settings)
{
    os << "Settings="
       << "["
       << "FPMixedPrecision=" << settings.mixed_precision() << "]";
    return os;
}

/** Formatted output of the arm_compute::experimental::dynamic_fusion::GpuPool2dSettings type.
 *
 * @param[in] settings arm_compute::experimental::dynamic_fusion::GpuPool2dSettings type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const experimental::dynamic_fusion::GpuPool2dSettings &settings)
{
    std::stringstream str;
    str << settings;
    return str.str();
}

/** Formatted output of the arm_compute::experimental::dynamic_fusion::Conv2dAttributes type.
 *
 * @param[out] os          Output stream.
 * @param[in]  conv2d_attr arm_compute::experimental::dynamic_fusion::Conv2dAttributes type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const experimental::dynamic_fusion::Conv2dAttributes &conv2d_attr)
{
    os << "Conv2dAttributes="
       << "["
       << "Padding=" << conv2d_attr.pad() << ", "
       << "Size2D=" << conv2d_attr.stride() << ", "
       << "Dialation=" << conv2d_attr.dilation() << "]";

    return os;
}

/** Formatted output of the arm_compute::experimental::dynamic_fusion::Conv2dAttributes type.
 *
 * @param[in] conv2d_attr arm_compute::experimental::dynamic_fusion::Conv2dAttributes type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const experimental::dynamic_fusion::Conv2dAttributes &conv2d_attr)
{
    std::stringstream str;
    str << conv2d_attr;
    return str.str();
}

/** Formatted output of the arm_compute::experimental::dynamic_fusion::CastAttributes type.
 *
 * @param[out] os        Output stream.
 * @param[in]  cast_attr arm_compute::experimental::dynamic_fusion::CastAttributes type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const experimental::dynamic_fusion::CastAttributes &cast_attr)
{
    os << "CastAttributes="
       << "["
       << "Data Type=" << cast_attr.data_type() << ", "
       << "Convert Policy=" << cast_attr.convert_policy() << "]";

    return os;
}
/** Formatted output of the arm_compute::experimental::dynamic_fusion::CastAttributes type.
 *
 * @param[in] cast_attr arm_compute::experimental::dynamic_fusion::CastAttributes type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const experimental::dynamic_fusion::CastAttributes &cast_attr)
{
    std::stringstream str;
    str << cast_attr;
    return str.str();
}

/** Formatted output of the arm_compute::experimental::dynamic_fusion::DepthwiseConv2dAttributes type.
 *
 * @param[out] os             Output stream.
 * @param[in]  dw_conv2d_attr arm_compute::experimental::dynamic_fusion::DepthwiseConv2dAttributes type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const experimental::dynamic_fusion::DepthwiseConv2dAttributes &dw_conv2d_attr)
{
    os << "DepthwiseConv2dAttributes="
       << "["
       << "Padding=" << dw_conv2d_attr.pad() << ", "
       << "Size2D=" << dw_conv2d_attr.stride() << ", "
       << "Depth Multiplier=" << dw_conv2d_attr.depth_multiplier() << ", "
       << "Dilation=" << dw_conv2d_attr.dilation() << ","
       << "DimensionRoundingType: " << dw_conv2d_attr.dimension_rounding_type() << "]";

    return os;
}
/** Formatted output of the arm_compute::experimental::dynamic_fusion::DepthwiseConv2dAttributes type.
 *
 * @param[in] dw_conv2d_attr arm_compute::experimental::dynamic_fusion::DepthwiseConv2dAttributes type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const experimental::dynamic_fusion::DepthwiseConv2dAttributes &dw_conv2d_attr)
{
    std::stringstream str;
    str << dw_conv2d_attr;
    return str.str();
}

/** Formatted output of the arm_compute::experimental::dynamic_fusion::ClampAttributes type.
 *
 * @param[out] os         Output stream.
 * @param[in]  clamp_attr arm_compute::experimental::dynamic_fusion::ClampAttributes type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const experimental::dynamic_fusion::ClampAttributes &clamp_attr)
{
    os << "ClampAttributes="
       << "["
       << "Min value=" << clamp_attr.min_val() << ", "
       << "Max value=" << clamp_attr.max_val() << "]";
    return os;
}
/** Formatted output of the arm_compute::experimental::dynamic_fusion::ClampAttributes type.
 *
 * @param[in] clamp_attr arm_compute::experimental::dynamic_fusion::ClampAttributes type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const experimental::dynamic_fusion::ClampAttributes &clamp_attr)
{
    std::stringstream str;
    str << clamp_attr;
    return str.str();
}

/** Formatted output of the arm_compute::experimental::dynamic_fusion::ResizeAttributes type.
 *
 * @param[out] os          Output stream.
 * @param[in]  resize_attr arm_compute::experimental::dynamic_fusion::ResizeAttributes type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const experimental::dynamic_fusion::ResizeAttributes &resize_attr)
{
    os << "ResizeAttributes="
       << "["
       << "AlignCorners=" << resize_attr.align_corners() << ", "
       << "InterpolationPolicy=" << resize_attr.interpolation_policy() << ", "
       << "OutputHeight=" << resize_attr.output_height() << ", "
       << "OutputWidth=" << resize_attr.output_width() << ", "
       << "SamplingPolicy=" << resize_attr.sampling_policy() << "]";
    return os;
}

/** Formatted output of the arm_compute::experimental::dynamic_fusion::ResizeAttributes type.
 *
 * @param[in] resize_attr arm_compute::experimental::dynamic_fusion::ResizeAttributes type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const experimental::dynamic_fusion::ResizeAttributes &resize_attr)
{
    std::stringstream str;
    str << resize_attr;
    return str.str();
}

/** Formatted output of the arm_compute::experimental::dynamic_fusion::SoftmaxAttributes type.
 *
 * @param[out] os           Output stream.
 * @param[in]  softmax_attr arm_compute::experimental::dynamic_fusion::SoftmaxAttributes type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const experimental::dynamic_fusion::SoftmaxAttributes &softmax_attr)
{
    os << "SoftmaxAttributes="
       << "["
       << "Beta=" << softmax_attr.beta() << ", "
       << "Is Log Softmax=" << softmax_attr.is_log_softmax() << ", "
       << "Axis=" << softmax_attr.axis() << "]";
    return os;
}
/** Formatted output of the arm_compute::experimental::dynamic_fusion::SoftmaxAttributes type.
 *
 * @param[in] softmax_attr arm_compute::experimental::dynamic_fusion::SoftmaxAttributes type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const experimental::dynamic_fusion::SoftmaxAttributes &softmax_attr)
{
    std::stringstream str;
    str << softmax_attr;
    return str.str();
}
/** Formatted output of the arm_compute::MatMulInfo type.
 *
 * @param[out] os          Output stream.
 * @param[in]  matmul_info arm_compute::MatMulInfo  type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const arm_compute::MatMulInfo &matmul_info)
{
    os << "MatMulKernelInfo="
       << "["
       << "adj_lhs=" << matmul_info.adj_lhs() << ", "
       << "adj_rhs=" << matmul_info.adj_rhs() << ", "
       << "fused_activation=" << matmul_info.fused_activation().activation() << "]";

    return os;
}
/** Formatted output of the arm_compute::MatMulInfo type.
 *
 * @param[in] matmul_info arm_compute::MatMulInfo type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const arm_compute::MatMulInfo &matmul_info)
{
    std::stringstream str;
    str << matmul_info;
    return str.str();
}

/** Formatted output of the arm_compute::MatMulKernelInfo type.
 *
 * @param[out] os          Output stream.
 * @param[in]  matmul_info arm_compute::MatMulKernelInfo  type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const arm_compute::MatMulKernelInfo &matmul_info)
{
    os << "MatMulKernelInfo="
       << "["
       << "adj_lhs=" << matmul_info.adj_lhs << ", "
       << "adj_rhs=" << matmul_info.adj_rhs << ", "
       << "M0=" << matmul_info.m0 << ", "
       << "N0=" << matmul_info.n0 << ", "
       << "K0=" << matmul_info.k0 << ", "
       << "export_rhs_to_cl_image=" << matmul_info.export_rhs_to_cl_image
       << "]";

    return os;
}
/** Formatted output of the arm_compute::MatMulKernelInfo type.
 *
 * @param[in] matmul_info arm_compute::MatMulKernelInfo type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const arm_compute::MatMulKernelInfo &matmul_info)
{
    std::stringstream str;
    str << matmul_info;
    return str.str();
}

/** Formatted output of the arm_compute::CpuMatMulSettings type.
 *
 * @param[out] os       Output stream.
 * @param[in]  settings arm_compute::CpuMatMulSettings type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const arm_compute::CpuMatMulSettings &settings)
{
    os << "CpuMatMulSettings="
       << "["
       << "fast_math=" << settings.fast_math()
       << "]";

    return os;
}
/** Formatted output of the arm_compute::CpuMatMulSettings type.
 *
 * @param[in] settings arm_compute::CpuMatMulSettings type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const arm_compute::CpuMatMulSettings &settings)
{
    std::stringstream str;
    str << settings;
    return str.str();
}

} // namespace arm_compute

#endif /* __ARM_COMPUTE_TYPE_PRINTER_H__ */
