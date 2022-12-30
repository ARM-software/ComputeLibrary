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
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuDepthwiseConv2d.h"

#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "src/common/utils/Log.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/dynamic_fusion/sketch/ArgumentPack.h"
#include "src/dynamic_fusion/sketch/gpu/GpuWorkloadSketchImpl.h"
#include "src/dynamic_fusion/sketch/gpu/components/cl/ClComponentDepthwiseConv2d.h"
#include "src/gpu/cl/kernels/gemm/ClGemmHelpers.h"
#include "src/runtime/heuristics/dwc_native/ClDWCNativeKernelConfig.h"
#include "src/runtime/heuristics/dwc_native/IClDWCNativeKernelConfig.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
namespace
{
void calculate_and_init_dst_if_empty(ITensorInfo *dst, const ITensorInfo *src, const ITensorInfo *wei, const DepthwiseConv2dAttributes &attributes)
{
    if(dst->total_size() == 0U)
    {
        const PadStrideInfo pad_stride_info(attributes.stride().x(),
                                            attributes.stride().y(),
                                            attributes.pad().left,
                                            attributes.pad().right,
                                            attributes.pad().top,
                                            attributes.pad().bottom,
                                            attributes.dimension_rounding_type());

        const ConvolutionInfo conv_info{ pad_stride_info, attributes.depth_multiplier(), ActivationLayerInfo(), attributes.dilation() };
        const TensorShape     shape = misc::shape_calculator::compute_depthwise_convolution_shape(*src, *wei, conv_info);

        auto_init_if_empty(*dst, src->clone()->set_tensor_shape(shape));
    }
}

constexpr GpuOperatorType operator_type = GpuOperatorType::Complex;
} // namespace

Status GpuDepthwiseConv2d::is_supported_op(const GpuWorkloadContext        &context,
                                           const ITensorInfo               *src,
                                           const ITensorInfo               *wei,
                                           const ITensorInfo               *bia,
                                           const ITensorInfo               *dst,
                                           const DepthwiseConv2dAttributes &attributes)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, wei, dst);

    // Auto initialize dst tensor info
    TensorInfo dst_info_to_validate = *dst;
    calculate_and_init_dst_if_empty(&dst_info_to_validate, src, wei, attributes);

    // Check support level
    // Data type
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::F16, DataType::F32);
    // Data layout
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(src, DataLayout::NHWC);

    const GpuTarget gpu_target = context.gpu_target();

    if(context.gpu_language() == GpuLanguage::OpenCL)
    {
        const CLCompileContext *cl_compile_ctx = context.cl_compile_context();
        ARM_COMPUTE_RETURN_ERROR_ON(cl_compile_ctx == nullptr);

        // Validate Depthwise Conv2d Component
        {
            const auto properties = IGpuKernelComponent::Properties().stage(UnitWorkloadStage{ UnitWorkloadStage::Stage::Run });
            auto       settings   = ClComponentDepthwiseConv2d::Settings();

            const PadStrideInfo legacy_conv_info(attributes.stride().x(), attributes.stride().y(), attributes.pad().left,
                                                 attributes.pad().right,
                                                 attributes.pad().top, attributes.pad().bottom, DimensionRoundingType::FLOOR);

            // Get the depthwise convolution compute parameters
            auto t = arm_compute::cl_dwc::ClDWCNativeKernelConfigurationFactory::create(gpu_target);
            const DWCComputeKernelInfo dwc_info = t->configure(src, wei, legacy_conv_info, attributes.dilation(), attributes.depth_multiplier());

            settings.fast_relaxed_math(
                (gpu_target != GPUTarget::G71 && (gpu_target & GPUTarget::GPU_ARCH_MASK) == GPUTarget::BIFROST)
                && (dst_info_to_validate.data_type() == DataType::F32 || dst_info_to_validate.data_type() == DataType::F16));

            settings.is_fma_available(get_arch_from_target(gpu_target) == GPUTarget::MIDGARD)
            .m0(dwc_info.m0)
            .n0(dwc_info.n0)
            .export_input_to_cl_image(dwc_info.export_input_to_cl_image)
            .export_weights_to_cl_image(dwc_info.export_weights_to_cl_image);

            ArgumentPack<ITensorInfo> arguments;
            arguments.add_const_tensor(ACL_SRC_0, src);
            arguments.add_const_tensor(ACL_SRC_1, wei);
            arguments.add_const_tensor(ACL_SRC_2, bia);
            arguments.add_const_tensor(ACL_DST_0, &dst_info_to_validate);
            ARM_COMPUTE_RETURN_ON_ERROR(ClComponentDepthwiseConv2d::validate(properties, arguments, attributes, settings));
        }
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_MSG("Unimplemented Gpu language");
    }

    return Status{};
}

Status GpuDepthwiseConv2d::validate_op(const GpuWorkloadSketch         &sketch,
                                       const ITensorInfo               *src,
                                       const ITensorInfo               *wei,
                                       const ITensorInfo               *bia,
                                       const ITensorInfo               *dst,
                                       const DepthwiseConv2dAttributes &attributes)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, wei, dst);
    ARM_COMPUTE_RETURN_ERROR_ON(!src->has_valid_id() || !wei->has_valid_id() || !dst->has_valid_id());

    if(bia != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(!bia->has_valid_id());
    }

    // Auto initialize dst tensor info
    TensorInfo dst_info_to_validate = *dst;
    calculate_and_init_dst_if_empty(&dst_info_to_validate, src, wei, attributes);

    // Perform fusion test
    // Pack tensor infos
    ArgumentPack<ITensorInfo> tensors;
    tensors.add_const_tensor(ACL_SRC_0, src);
    tensors.add_const_tensor(ACL_SRC_1, wei);
    tensors.add_const_tensor(ACL_SRC_2, bia);
    tensors.add_const_tensor(ACL_DST_0, &dst_info_to_validate);
    const Operator op = sketch.implementation().operator_group().new_operator(operator_type, tensors);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!sketch.implementation().operator_group().try_add_operator(op),
                                    "Operator fusion test failed. This operator cannot be fused into the workload");

    // Check if configuration is supported
    return is_supported_op(*sketch.gpu_context(), src, wei, bia, &dst_info_to_validate, attributes);
}

void GpuDepthwiseConv2d::create_op(GpuWorkloadSketch               &sketch,
                                   ITensorInfo                     *src,
                                   ITensorInfo                     *wei,
                                   ITensorInfo                     *bia,
                                   ITensorInfo                     *dst,
                                   const DepthwiseConv2dAttributes &attributes)
{
    // Assert validation
    ARM_COMPUTE_ERROR_THROW_ON(GpuDepthwiseConv2d::validate_op(sketch, src, wei, bia, dst, attributes));
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, wei, dst);
    ARM_COMPUTE_LOG_PARAMS(src, wei, bia, dst, attributes);

    calculate_and_init_dst_if_empty(dst, src, wei, attributes);

    // Translate into components and add to component graph
    GpuKernelComponentGraph &comp_graph = sketch.implementation().component_graph();
    const auto              *sketch_ctx = sketch.implementation().context();
    const GpuTarget          gpu_target = sketch_ctx->gpu_target();

    if(sketch_ctx->gpu_language() == GpuLanguage::OpenCL)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(sketch_ctx->cl_compile_context());

        // Add Depthwise Conv2d Component
        {
            const auto properties = IGpuKernelComponent::Properties().stage(UnitWorkloadStage{ UnitWorkloadStage::Stage::Run });
            auto       settings   = ClComponentDepthwiseConv2d::Settings();

            const PadStrideInfo legacy_conv_info(attributes.stride().x(), attributes.stride().y(), attributes.pad().left,
                                                 attributes.pad().right,
                                                 attributes.pad().top, attributes.pad().bottom, DimensionRoundingType::FLOOR);

            // Get the depthwise convolution compute parameters
            auto t = arm_compute::cl_dwc::ClDWCNativeKernelConfigurationFactory::create(gpu_target);
            const DWCComputeKernelInfo dwc_info = t->configure(src, wei, legacy_conv_info, attributes.dilation(), attributes.depth_multiplier());

            settings.is_fma_available(get_arch_from_target(gpu_target) != GPUTarget::MIDGARD)
            .m0(dwc_info.m0)
            .n0(dwc_info.n0)
            .export_input_to_cl_image(dwc_info.export_input_to_cl_image)
            .export_weights_to_cl_image(dwc_info.export_weights_to_cl_image);

            if(settings.export_input_to_cl_image())
            {
                arm_compute::opencl::kernels::gemm::update_padding_for_cl_image(src);
            }

            if(settings.export_weights_to_cl_image())
            {
                arm_compute::opencl::kernels::gemm::update_padding_for_cl_image(wei);
            }

            ArgumentPack<ITensorInfo> arguments;
            arguments.add_const_tensor(ACL_SRC_0, src);
            arguments.add_const_tensor(ACL_SRC_1, wei);
            arguments.add_const_tensor(ACL_SRC_2, bia);
            arguments.add_const_tensor(ACL_DST_0, dst);
            comp_graph.add_new_component<ClComponentDepthwiseConv2d>(properties, arguments, attributes, settings);
        }
    }
    else
    {
        ARM_COMPUTE_ERROR("Unimplemented Gpu language");
    }

    // Set up fusion test by adding to the Operator Group
    // Note this has to be performed after all the components have been successfully added to the component graph

    // Pack tensor infos
    ArgumentPack<ITensorInfo> tensors;
    tensors.add_const_tensor(ACL_SRC_0, src);
    tensors.add_const_tensor(ACL_SRC_1, wei);
    tensors.add_const_tensor(ACL_SRC_2, bia);
    tensors.add_const_tensor(ACL_DST_0, dst);

    const Operator op = sketch.implementation().operator_group().new_operator(operator_type, tensors);
    sketch.implementation().operator_group().add_operator(op);
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
