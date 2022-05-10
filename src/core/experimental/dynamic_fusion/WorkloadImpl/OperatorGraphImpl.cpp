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
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include "src/core/experimental/dynamic_fusion/WorkloadImpl/ClKernelGraph.h"
#include "src/core/experimental/dynamic_fusion/WorkloadImpl/OperatorGraphImpl.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
namespace
{
Status add_kernel_tensor(ClKernelGraph &k_graph, const OperatorGraph::Implementation &op_graph, const OpTensorContent &op_tensor, MemoryType memory_type, AuxMemoryInfo memory_info,
                         DependencyGraph::Id &id)
{
    ARM_COMPUTE_UNUSED(op_graph);
    return k_graph.add_kernel_tensor(op_tensor.desc, memory_type, memory_info, id, op_tensor.id);
}

Status add_kernel_tensor(ClKernelGraph &k_graph, const OperatorGraph::Implementation &op_graph, const OpTensorContent &op_tensor, DependencyGraph::Id &id)
{
    // For a tensor t
    // 1. If t is a src tensor of the entire op graph, then it's Core.
    //    (Optimisation opportunity, if we guanrantee that all translate methods are called in topological order, we can always assign t to Core.
    //       Because even if the op is non-root (which would mean t should be an Aux tensor), the src tensors would be already be determined by the ancestor ops (topological order), and thus would not be overriden by it)
    // 2. If t is a dst tensor of the entire op graph, then it's Core.
    // 3. Aux tensor with Persistent and Prepare lifetime is manually specified
    // 4. All other ts not captured by the above are assigned Aux, with lifetime of Temporary.
    // kernel_graph.add_kernel_tensor(input->desc, );
    bool          is_src_tensor_of_graph = is_in(op_tensor.id, op_graph.graph.src_tensors());
    bool          is_dst_tensor_of_graph = is_in(op_tensor.id, op_graph.graph.dst_tensors());
    MemoryType    memory_type;
    AuxMemoryInfo memory_info;
    if(is_src_tensor_of_graph || is_dst_tensor_of_graph)
    {
        memory_type = MemoryType::Core;
    }
    else
    {
        memory_type          = MemoryType::Auxiliary;
        memory_info.lifetime = AuxMemoryLifetime::Temporary;
        memory_info.size     = op_tensor.desc->total_size();
    }
    return add_kernel_tensor(k_graph, op_graph, op_tensor, memory_type, memory_info, id);
}

/** Get the suitable kernel size for using direct convolution method with NHWC data layout.
 *
 * @note Duplicate of the function with the same name in src/gpu/cl/operators/ClConv2d.cpp
 *
 * @note Direct convolution should be executed when the kernel has the spatial dimensions greater than or equal to the value returned by this function
 *
 * @param[in] gpu_target GPU target
 *
 * @return the suitable kernel size for using direct convolution method with NHWC data layout
 */
size_t get_direct_conv_kernel_threshold_nhwc(arm_compute::GPUTarget gpu_target)
{
    switch(gpu_target)
    {
        case arm_compute::GPUTarget::G76:
        case arm_compute::GPUTarget::G77:
        case arm_compute::GPUTarget::G78:
            return 5;
        case arm_compute::GPUTarget::G71:
        case arm_compute::GPUTarget::G72:
        case arm_compute::GPUTarget::MIDGARD:
        case arm_compute::GPUTarget::BIFROST:
            return 7;
        default:
            return 5;
    }
}
} // namespace

bool operator==(const OpTensor &t0, const OpTensor &t1)
{
    return std::make_tuple(t0.id()) == std::make_tuple(t1.id());
}
bool operator==(const Padding2D &pad0, const Padding2D &pad1)
{
    return std::make_tuple(pad0.top, pad0.right, pad0.bottom, pad0.left) == std::make_tuple(pad1.top, pad1.right, pad1.bottom, pad1.left);
}
bool operator==(const Conv2dDescriptor &conv2d0, const Conv2dDescriptor &conv2d1)
{
    return std::make_tuple(conv2d0.pad, conv2d0.stride, conv2d0.dilation) == std::make_tuple(conv2d1.pad, conv2d1.stride, conv2d1.dilation);
}

bool operator==(const AddDescriptor &, const AddDescriptor &)
{
    return std::make_tuple() == std::make_tuple(); // Currently two Add ops are always the same
}

bool Conv2dContent::operator==(const OperatorContent &other) const
{
    const auto converted = *utils::cast::polymorphic_downcast<const Conv2dContent *>(&other);
    return desc == converted.desc;
}

bool AddContent::operator==(const OperatorContent &other) const
{
    const auto converted = *utils::cast::polymorphic_downcast<const AddContent *>(&other);
    return desc == converted.desc;
}

ConvolutionMethod Conv2dContent::select_conv_method(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *dst, const Conv2dDescriptor &conv2d_desc, const GPUTarget gpu_target)
{
    // Modified from ClConv2d::get_convolution_method

    ARM_COMPUTE_ERROR_ON_NULLPTR(src);
    ARM_COMPUTE_ERROR_ON_NULLPTR(dst);
    ARM_COMPUTE_ERROR_ON_NULLPTR(weights);

    const PadStrideInfo legacy_pad_stride(conv2d_desc.stride.x(), conv2d_desc.stride.y(), conv2d_desc.pad.left, conv2d_desc.pad.right, conv2d_desc.pad.top, conv2d_desc.pad.bottom, DimensionRoundingType{});
    const Size2D        dilation = conv2d_desc.dilation;

    const size_t idx_w = get_data_layout_dimension_index(src->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_h = get_data_layout_dimension_index(src->data_layout(), DataLayoutDimension::HEIGHT);
    const size_t idx_c = get_data_layout_dimension_index(src->data_layout(), DataLayoutDimension::CHANNEL);

    /* Input spatial dims, kernel size, IFM/OFM, conv info*/
    using ConvolutionConfiguration = std::tuple<Size2D, Size2D, Size2D, PadStrideInfo, DataLayout>;
    using ConfigurationMethod      = std::pair<ConvolutionConfiguration, ConvolutionMethod>;

    const std::vector<ConfigurationMethod> known_configs =
    {
        // Alexnet
        ConfigurationMethod(ConvolutionConfiguration(Size2D(27U, 27U), Size2D(5U, 5U), Size2D(48U, 128U), PadStrideInfo(1U, 1U, 2U, 2U), DataLayout::NCHW), ConvolutionMethod::DIRECT),
        // VGG16 / VGG19
        ConfigurationMethod(ConvolutionConfiguration(Size2D(224U, 224U), Size2D(3U, 3U), Size2D(3U, 64U), PadStrideInfo(1U, 1U, 1U, 1U), DataLayout::NCHW), ConvolutionMethod::DIRECT),
        // Mobilenet 224
        ConfigurationMethod(ConvolutionConfiguration(Size2D(224U, 224U), Size2D(3U, 3U), Size2D(3U, 32U), PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR), DataLayout::NCHW), ConvolutionMethod::GEMM),
        // Mobilenet 160
        ConfigurationMethod(ConvolutionConfiguration(Size2D(160U, 160U), Size2D(3U, 3U), Size2D(3U, 24U), PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR), DataLayout::NCHW), ConvolutionMethod::GEMM),
        // Mobilenet 224
        ConfigurationMethod(ConvolutionConfiguration(Size2D(224U, 224U), Size2D(3U, 3U), Size2D(3U, 32U), PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR), DataLayout::NHWC), ConvolutionMethod::GEMM),
        // Mobilenet 160
        ConfigurationMethod(ConvolutionConfiguration(Size2D(160U, 160U), Size2D(3U, 3U), Size2D(3U, 24U), PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR), DataLayout::NHWC), ConvolutionMethod::GEMM),
    };

    const auto find_config = [&](ConfigurationMethod c)
    {
        const ConvolutionConfiguration config      = c.first;
        const PadStrideInfo            info        = std::get<3>(config);
        const DataLayout               data_layout = std::get<4>(config);

        return std::get<0>(config) == Size2D(src->dimension(idx_w), src->dimension(idx_h)) && std::get<1>(config) == Size2D(weights->dimension(idx_w), weights->dimension(idx_h))
               && std::get<2>(config) == Size2D(weights->dimension(idx_c), weights->dimension(3)) && info.pad_top() == legacy_pad_stride.pad_top() && info.pad_right() == legacy_pad_stride.pad_right()
               && info.pad_bottom() == legacy_pad_stride.pad_bottom() && info.pad_left() == legacy_pad_stride.pad_left() && info.stride() == legacy_pad_stride.stride() && (data_layout == src->data_layout());
    };

    std::vector<ConfigurationMethod>::const_iterator found;
    if((found = std::find_if(known_configs.begin(), known_configs.end(), find_config)) != known_configs.end())
    {
        return (*found).second;
    }

    if(dilation != Size2D(1U, 1U))
    {
        return ConvolutionMethod::GEMM;
    }
    else
    {
        if(src->data_layout() == DataLayout::NCHW)
        {
            ARM_COMPUTE_ERROR("NCHW not supported");
        }
        else
        {
            const bool   is_direct_valid           = bool(ClDirectConv2dKernel::validate(src, weights, nullptr, dst, ClDirectConv2dKernelDescriptor{ conv2d_desc }));
            const size_t kernel_sz_direct_conv_thr = get_direct_conv_kernel_threshold_nhwc(gpu_target);

            // SRGAN case
            if((src->dimension(idx_h) > 720U) && (dst->dimension(idx_h) > 720U) && (weights->dimension(idx_h) == 9) && (conv2d_desc.pad.top < 3)
               && is_direct_valid)
            {
                return ConvolutionMethod::DIRECT;
            }

            // Floating-point case: GeMM/Direct
            if(is_data_type_float(src->data_type()))
            {
                // Get dst shape
                TensorShape output_shape       = misc::shape_calculator::compute_deep_convolution_shape(*src, *weights, legacy_pad_stride);
                const bool  is_large_kernel_sz = (weights->dimension(idx_w) >= kernel_sz_direct_conv_thr) && (weights->dimension(idx_h) >= kernel_sz_direct_conv_thr);
                const bool  is_ifm_ge_16       = src->dimension(idx_c) >= 16;
                const bool  is_ofm_lte_8       = weights->dimension(3U) <= 8;
                const bool  workload_gte_8192  = (output_shape[0] * output_shape[1] * output_shape[2]) / 16 >= 8192;
                const bool  is_ifm_gt_ofm      = src->dimension(idx_c) > weights->dimension(3U);

                // Direct convolution case
                if(is_direct_valid)
                {
                    if((gpu_target == arm_compute::GPUTarget::G71 || gpu_target == arm_compute::GPUTarget::G72 || gpu_target == arm_compute::GPUTarget::MIDGARD))
                    {
                        if(is_large_kernel_sz && is_ifm_ge_16 && is_ifm_gt_ofm)
                        {
                            return ConvolutionMethod::DIRECT;
                        }
                    }
                    else
                    {
                        if((is_large_kernel_sz && workload_gte_8192 && is_ifm_ge_16) || (is_ofm_lte_8 && is_ifm_ge_16))
                        {
                            return ConvolutionMethod::DIRECT;
                        }
                    }
                }

                // Default case
                return ConvolutionMethod::GEMM;
            }

            // Generic case for quantized. Only GeMM
            return ConvolutionMethod::GEMM;
        }
    }
    return ConvolutionMethod::DIRECT;
}

Status Conv2dContent::translate(ClKernelGraph &kernel_graph) const
{
    const auto input  = _tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const auto weight = _tensors.get_const_tensor(TensorType::ACL_SRC_1);
    const auto dst    = _tensors.get_const_tensor(TensorType::ACL_DST_0);
    const auto method = forced_method_enabled ? forced_method : Conv2dContent::select_conv_method(input->desc, weight->desc, dst->desc, desc, CLScheduler::get().target());
    switch(method)
    {
        case ConvolutionMethod::DIRECT:
        {
            return translate_direct_conv2d(kernel_graph);
        }
        default:
        {
            ARM_COMPUTE_RETURN_ERROR_MSG("Not implemented");
        }
    }
    return Status{};
}
Status Conv2dContent::translate_direct_conv2d(ClKernelGraph &kernel_graph) const
{
    const auto input  = _tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const auto weight = _tensors.get_const_tensor(TensorType::ACL_SRC_1);
    const auto bias   = _tensors.get_const_tensor(TensorType::ACL_SRC_2);
    const auto dst    = _tensors.get_const_tensor(TensorType::ACL_DST_0);
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weight, dst);

    ITensorDescPack<ClKernelTensor> tensors;

    DependencyGraph::Id input_id;
    auto                st = add_kernel_tensor(kernel_graph, *_graph, *input, input_id);
    ARM_COMPUTE_RETURN_ON_ERROR(st);
    tensors.add_const_tensor(ACL_SRC_0, kernel_graph.get_tensor(input_id));

    DependencyGraph::Id weight_id;
    st = add_kernel_tensor(kernel_graph, *_graph, *weight, weight_id);
    ARM_COMPUTE_RETURN_ON_ERROR(st);
    tensors.add_const_tensor(ACL_SRC_1, kernel_graph.get_tensor(weight_id));

    if(bias != nullptr)
    {
        DependencyGraph::Id bias_id;
        st = add_kernel_tensor(kernel_graph, *_graph, *bias, bias_id);
        ARM_COMPUTE_RETURN_ON_ERROR(st);
        tensors.add_const_tensor(ACL_SRC_2, kernel_graph.get_tensor(bias_id));
    }

    DependencyGraph::Id dst_id;
    st = add_kernel_tensor(kernel_graph, *_graph, *dst, dst_id);
    ARM_COMPUTE_RETURN_ON_ERROR(st);
    tensors.add_const_tensor(ACL_DST_0, kernel_graph.get_tensor(dst_id));

    DependencyGraph::Id direct_conv2d_id;
    const auto          kernel_desc = ClDirectConv2dKernelDescriptor{ desc };

    st = ClDirectConv2dKernel::validate(input->desc, weight->desc, bias == nullptr ? nullptr : bias->desc, dst->desc, kernel_desc);
    ARM_COMPUTE_RETURN_ON_ERROR(st);

    ClKernelConfig config{ UnitWorkloadStage{ UnitWorkloadStage::Stage::Run }, TileDescriptor{}, StoreType::TStoreIndirectWidthSelect };
    st = kernel_graph.add_kernel<ClDirectConv2dKernel>(config, kernel_desc, tensors, direct_conv2d_id);
    ARM_COMPUTE_RETURN_ON_ERROR(st);
    ARM_COMPUTE_UNUSED(direct_conv2d_id);

    return Status{};
}

Status AddContent::translate(ClKernelGraph &kernel_graph) const
{
    const auto lhs = _tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const auto rhs = _tensors.get_const_tensor(TensorType::ACL_SRC_1);
    const auto dst = _tensors.get_const_tensor(TensorType::ACL_DST_0);
    ARM_COMPUTE_ERROR_ON_NULLPTR(lhs, rhs, dst);

    ITensorDescPack<ClKernelTensor> tensors;

    DependencyGraph::Id lhs_id;
    auto                st = add_kernel_tensor(kernel_graph, *_graph, *lhs, lhs_id);
    ARM_COMPUTE_RETURN_ON_ERROR(st);
    tensors.add_const_tensor(ACL_SRC_0, kernel_graph.get_tensor(lhs_id));

    DependencyGraph::Id rhs_id;
    st = add_kernel_tensor(kernel_graph, *_graph, *rhs, rhs_id);
    ARM_COMPUTE_RETURN_ON_ERROR(st);
    tensors.add_const_tensor(ACL_SRC_1, kernel_graph.get_tensor(rhs_id));

    DependencyGraph::Id dst_id;
    st = add_kernel_tensor(kernel_graph, *_graph, *dst, dst_id);
    ARM_COMPUTE_RETURN_ON_ERROR(st);
    tensors.add_const_tensor(ACL_DST_0, kernel_graph.get_tensor(dst_id));

    DependencyGraph::Id add_id;
    ClKernelConfig      config{ UnitWorkloadStage{ UnitWorkloadStage::Stage::Run }, TileDescriptor{}, StoreType::TStoreIndirectWidthSelect };

    st = ClAddKernel::validate(lhs->desc, rhs->desc, dst->desc);
    ARM_COMPUTE_RETURN_ON_ERROR(st);

    st = kernel_graph.add_kernel<ClAddKernel>(config, ClEltwiseAddKernelDescriptor{ desc }, tensors, add_id);
    ARM_COMPUTE_RETURN_ON_ERROR(st);
    ARM_COMPUTE_UNUSED(add_id);

    return Status{};
}

std::vector<const OperatorContent *> traverse(const OperatorGraph::Implementation &graph)
{
    std::vector<const OperatorContent *> ops;
    const auto                           sorted = graph.graph.topological_sort();
    for(const auto &pack : sorted.second)
    {
        ops.push_back(graph.operators.at(pack.op).get());
    }
    return ops;
}

std::vector<OperatorContent *> traverse(OperatorGraph::Implementation &graph)
{
    std::vector<OperatorContent *> ops;
    const auto                     sorted = graph.graph.topological_sort();
    for(const auto &pack : sorted.second)
    {
        ops.push_back(graph.operators.at(pack.op).get());
    }
    return ops;
}

Status translate(ClKernelGraph &kernel_graph, const OperatorGraph::Implementation &op_graph)
{
    for(const auto &op : traverse(op_graph))
    {
        const auto st = op->translate(kernel_graph);
        ARM_COMPUTE_RETURN_ON_ERROR(st);
    }
    return Status{};
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */