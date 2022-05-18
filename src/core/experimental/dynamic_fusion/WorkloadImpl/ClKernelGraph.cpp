/*
 * Copyright (c) 2022 Arm Limited.
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
#ifdef ENABLE_EXPERIMENTAL_DYNAMIC_FUSION
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "src/core/CL/CLValidate.h"
#include "src/core/experimental/dynamic_fusion/ClKernelBuildingAPI.h"
#include "src/core/experimental/dynamic_fusion/WorkloadImpl/ClKernelGraph.h"

#include "support/Cast.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
Status ClDirectConv2dKernel::generate(ClKernelBlueprint &bp) const
{
    const auto input  = _tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const auto weight = _tensors.get_const_tensor(TensorType::ACL_SRC_1);
    const auto bias   = _tensors.get_const_tensor(TensorType::ACL_SRC_2);
    const auto dst    = _tensors.get_const_tensor(TensorType::ACL_DST_0);
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weight, dst);
    ArgumentID input_id;
    add_tensor(bp, input->desc, input_id, input->id);
    ArgumentID weight_id;
    add_tensor(bp, weight->desc, weight_id, weight->id);
    ArgumentID bias_id = g_arg_placeholder;
    if(bias != nullptr)
    {
        add_tensor(bp, bias->desc, bias_id, bias->id);
    }
    ArgumentID dst_id;
    add_tensor(bp, dst->desc, dst_id, dst->id);

    add_kcomp_direct_conv2d(bp, desc, input_id, weight_id, bias_id, dst_id);
    return Status{};
}
Status ClDirectConv2dKernel::validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst, const ClDirectConv2dKernelDescriptor &conv2d_desc)
{
    // 1. Check validity
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, weights, dst);
    // Matching data type
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, weights);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
    if(biases != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, biases);
    }

    // Matching data layout
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(src, weights);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(src, dst);
    if(biases != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(src, biases);
    }

    // All tensor infos are initialized
    ARM_COMPUTE_RETURN_ERROR_ON(src->tensor_shape().total_size() == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(weights->tensor_shape().total_size() == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(dst->tensor_shape().total_size() == 0);
    if(biases != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(biases->tensor_shape().total_size() == 0);
    }
    // Device requirements are met
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src);
    // weights shape is correct
    const DataLayout data_layout = src->data_layout();
    const int        channel_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->dimension(channel_idx) != src->dimension(channel_idx), "Weights feature map dimension should match the respective src's one");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->num_dimensions() > 4, "Weights can be at most 4 dimensional");

    // dst shape is correct
    PadStrideInfo legacy_pad_stride(conv2d_desc.conv2d.stride.x(), conv2d_desc.conv2d.stride.y(), conv2d_desc.conv2d.pad.left, conv2d_desc.conv2d.pad.right, conv2d_desc.conv2d.pad.top,
                                    conv2d_desc.conv2d.pad.bottom, DimensionRoundingType{});
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(dst->tensor_shape(),
                                                       misc::shape_calculator::compute_deep_convolution_shape(*src, *weights, legacy_pad_stride));

    // biases shape is correct
    if(biases != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(biases->dimension(0) != weights->dimension(3),
                                        "Biases size and number of dst feature maps should match");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(biases->num_dimensions() > 1,
                                        "Biases should be one dimensional");
    }

    // 2. Check support level
    // Data type
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::F16, DataType::F32);
    // Data layout
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(src, DataLayout::NHWC);

    return Status{};
}

bool ClDirectConv2dKernel::operator==(const ClKernel &other) const
{
    const auto converted = *utils::cast::polymorphic_downcast<const ClDirectConv2dKernel *>(&other);
    return config() == other.config() && tensors() == other.tensors() && desc == converted.desc;
}

Status ClAddKernel::generate(ClKernelBlueprint &bp) const
{
    const auto lhs = _tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const auto rhs = _tensors.get_const_tensor(TensorType::ACL_SRC_1);
    const auto dst = _tensors.get_const_tensor(TensorType::ACL_DST_0);
    ARM_COMPUTE_ERROR_ON_NULLPTR(lhs, rhs, dst);
    ArgumentID lhs_id;
    add_tensor(bp, lhs->desc, lhs_id, lhs->id);
    ArgumentID rhs_id;
    add_tensor(bp, rhs->desc, rhs_id, rhs->id);
    ArgumentID dst_id;
    add_tensor(bp, dst->desc, dst_id, dst->id);

    add_kcomp_eltwise_add(bp, desc, lhs_id, rhs_id, dst_id);
    return Status{};
}

Status ClAddKernel::validate(const ITensorInfo *lhs, const ITensorInfo *rhs, const ITensorInfo *dst)
{
    // 1. Check validity
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(lhs, rhs, dst);

    // Matching data type
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(lhs, rhs);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(lhs, dst);

    // Matching data layout
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(lhs, rhs);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(lhs, dst);

    // All tensor infos are initialized
    ARM_COMPUTE_RETURN_ERROR_ON(lhs->tensor_shape().total_size() == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(rhs->tensor_shape().total_size() == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(dst->tensor_shape().total_size() == 0);

    // Device requirements are met
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(lhs);

    const bool in_place      = (lhs == dst) || (rhs == dst);
    const bool src0_in_place = in_place && (lhs == dst);

    // dst shape is correct
    const TensorShape out_shape = TensorShape::broadcast_shape(lhs->tensor_shape(), rhs->tensor_shape());
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(out_shape, dst->tensor_shape(), 0), "Wrong shape for dst");
    if(in_place)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(out_shape, src0_in_place ? lhs->tensor_shape() : rhs->tensor_shape(), 0),
                                        "Wrong shape for dst, cannot do in_place calculation");
    }

    // 2. Check support level

    // Data type
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(lhs, 1, DataType::F32, DataType::F16);

    // Data layout
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(lhs, DataLayout::NHWC);

    return Status{};
}

bool ClAddKernel::operator==(const ClKernel &other) const
{
    const auto converted = *utils::cast::polymorphic_downcast<const ClAddKernel *>(&other);
    return config() == other.config() && tensors() == other.tensors() && desc == converted.desc;
}

std::vector<const ClKernel *> traverse(const ClKernelGraph &graph)
{
    std::vector<const ClKernel *> kernels;
    const auto                    sorted = graph.graph.topological_sort();
    for(const auto &pack : sorted.second)
    {
        kernels.push_back(graph.kernels.at(pack.op).get());
    }
    return kernels;
}
std::vector<ClKernel *> traverse(ClKernelGraph &graph)
{
    std::vector<ClKernel *> kernels;
    const auto              sorted = graph.graph.topological_sort();
    for(const auto &pack : sorted.second)
    {
        kernels.push_back(graph.kernels.at(pack.op).get());
    }
    return kernels;
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */