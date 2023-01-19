/*
 * Copyright (c) 2022-2023 Arm Limited.
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
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuConv2d.h"

#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include "src/common/utils/Log.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/dynamic_fusion/sketch/ArgumentPack.h"
#include "src/dynamic_fusion/sketch/gpu/GpuWorkloadSketchImpl.h"
#include "src/dynamic_fusion/sketch/gpu/components/cl/ClComponentDirectConv2d.h"
#include "src/gpu/cl/kernels/gemm/ClGemmHelpers.h"
#include "src/runtime/heuristics/direct_conv/ClDirectConvKernelConfig.h"
#include "src/runtime/heuristics/direct_conv/IClDirectConvKernelConfig.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
namespace
{
bool export_to_cl_image_support(const ITensorInfo *tensor, GPUTarget gpu_target, const cl::Device &device, DataLayout data_layout)
{
    if(tensor->tensor_shape()[0] % 4 || (data_layout != DataLayout::NHWC))
    {
        return false;
    }

    // If not floating point
    if(!is_data_type_float(tensor->data_type()))
    {
        return false;
    }

    if(gpu_target == GPUTarget::G71 || get_arch_from_target(gpu_target) == GPUTarget::MIDGARD)
    {
        return false;
    }

    // Check if the cl_khr_image2d_from_buffer extension is supported on the target platform
    if(!image2d_from_buffer_supported(device))
    {
        return false;
    }

    // Check cl image pitch alignment
    if(get_cl_image_pitch_alignment(device) == 0)
    {
        return false;
    }

    const size_t image_w     = tensor->tensor_shape()[0] / 4;
    const size_t image_h     = tensor->tensor_shape()[1] * tensor->tensor_shape()[2] * tensor->tensor_shape()[3];
    const size_t max_image_w = device.getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>();
    const size_t max_image_h = device.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>();

    if(image_w > max_image_w || image_h > max_image_h)
    {
        return false;
    }

    return true;
}

DirectConvComputeKernelInfo config_direct_convolution_nhwc(const ITensorInfo *src, const ITensorInfo *weights, const PadStrideInfo &conv_info)
{
    // Get GPU target
    GPUTarget gpu_target = CLScheduler::get().target();

    std::unique_ptr<arm_compute::cl_direct_conv::IClDirectConvKernelConfig> t = arm_compute::cl_direct_conv::ClDirectConvKernelConfigurationFactory::create(gpu_target);

    return t->configure(src, weights, conv_info);
}

void calculate_and_init_dst_if_empty(ITensorInfo *dst, const ITensorInfo *src, const ITensorInfo *wei, const Conv2dAttributes &attributes)
{
    if(dst->total_size() == 0U)
    {
        const auto shape = misc::shape_calculator::compute_deep_convolution_shape(src->tensor_shape(), src->data_layout(), wei->tensor_shape(),
                                                                                  PadStrideInfo(attributes.stride().x(), attributes.stride().y(), attributes.pad().left,
                                                                                                attributes.pad().right,
                                                                                                attributes.pad().top, attributes.pad().bottom, DimensionRoundingType::FLOOR)); // use the default DimensionRoundingType

        auto_init_if_empty(*dst, src->clone()->set_tensor_shape(shape));
    }
}

/* A helper method to reduce the duplication in dst tensor initialization
*  when calling validate()
*/
Status is_supported_op_helper(const GpuWorkloadContext &context,
                              const ITensorInfo        *src,
                              const ITensorInfo        *wei,
                              const ITensorInfo        *bia,
                              const ITensorInfo        *dst,
                              const Conv2dAttributes   &attributes)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, wei);

    TensorInfo         dst_info_to_validate;
    const ITensorInfo *dst_info_to_validate_ptr = &dst_info_to_validate;

    const DataLayout data_layout = src->data_layout();
    if(dst != nullptr)
    {
        dst_info_to_validate_ptr = dst;
    }

    calculate_and_init_dst_if_empty(&dst_info_to_validate, src, wei, attributes);

    // Check support level
    // Data type
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::F16, DataType::F32);
    // Data layout
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(src, DataLayout::NHWC);

    // Check components
    const auto gpu_target = context.gpu_target();
    if(context.gpu_language() == GpuLanguage::OpenCL)
    {
        const auto cl_compile_ctx = context.cl_compile_context();
        ARM_COMPUTE_RETURN_ERROR_ON(cl_compile_ctx == nullptr);
        // Validate Direct Conv2d Component
        {
            const auto properties = IGpuKernelComponent::Properties().stage(UnitWorkloadStage{ UnitWorkloadStage::Stage::Run });
            auto       settings   = ClComponentDirectConv2d::Settings();

            settings.export_to_cl_image(
                export_to_cl_image_support(src, gpu_target, cl_compile_ctx->get_device(), data_layout));

            settings.fast_relaxed_math(
                (gpu_target != GPUTarget::G71 && (gpu_target & GPUTarget::GPU_ARCH_MASK) == GPUTarget::BIFROST)
                && (dst_info_to_validate_ptr->data_type() == DataType::F32 || dst_info_to_validate_ptr->data_type() == DataType::F16));

            ArgumentPack<ITensorInfo> arguments;
            arguments.add_const_tensor(ACL_SRC_0, src);
            arguments.add_const_tensor(ACL_SRC_1, wei);
            arguments.add_const_tensor(ACL_SRC_2, bia);
            arguments.add_const_tensor(ACL_DST_0, dst_info_to_validate_ptr);
            ARM_COMPUTE_RETURN_ON_ERROR(ClComponentDirectConv2d::validate(properties, arguments, attributes, settings));
        }
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_MSG("Unimplemented Gpu language");
    }
    return Status{};
}

constexpr GpuOperatorType operator_type = GpuOperatorType::Complex;
} // namespace

Status GpuConv2d::is_supported_op(const GpuWorkloadContext &context,
                                  const ITensorInfo        *src,
                                  const ITensorInfo        *wei,
                                  const ITensorInfo        *bia,
                                  const Conv2dAttributes   &attributes)
{
    return is_supported_op_helper(context, src, wei, bia, nullptr, attributes);
}

Status GpuConv2d::validate_op(const GpuWorkloadSketch &sketch,
                              const ITensorInfo       *src,
                              const ITensorInfo       *wei,
                              const ITensorInfo       *bia,
                              const Conv2dAttributes &attributes)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, wei);

    // Check if tensors have valid id. I.e. they are created from a sketch
    ARM_COMPUTE_RETURN_ERROR_ON(!src->has_valid_id() || !wei->has_valid_id());
    if(bia != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(!bia->has_valid_id());
    }

    // This tensor info will have invalid id but because all the existing tensors in the
    // sketch have valid ids and the DependencyGraph implementation has no notion of validness
    // regarding tensor ids, it'll be just another tensor id and will validate
    // Additionally, a new dst id is added every time in create_op, thus there's no need to validate it
    TensorInfo dst_info_to_validate;

    // Auto initialize dst tensor info
    calculate_and_init_dst_if_empty(&dst_info_to_validate, src, wei, attributes);

    // Perform fusion test
    // Check if operator meets fusion constraints
    ArgumentPack<ITensorInfo> tensors;
    tensors.add_const_tensor(ACL_SRC_0, src);
    tensors.add_const_tensor(ACL_SRC_1, wei);
    tensors.add_const_tensor(ACL_SRC_2, bia);
    tensors.add_const_tensor(ACL_DST_0, &dst_info_to_validate);
    const auto op = sketch.implementation().operator_group().new_operator(operator_type, tensors);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!sketch.implementation().operator_group().try_add_operator(op),
                                    "Operator fusion test failed. This operator cannot be fused into the workload");

    // Check if configuration is supported
    return is_supported_op_helper(*sketch.gpu_context(), src, wei, bia, &dst_info_to_validate, attributes);
}

ITensorInfo *GpuConv2d::create_op(GpuWorkloadSketch      &sketch,
                                  ITensorInfo            *src,
                                  ITensorInfo            *wei,
                                  ITensorInfo            *bia,
                                  const Conv2dAttributes &attributes)
{
    ARM_COMPUTE_LOG_PARAMS(src, wei, bia, attributes);
    PadStrideInfo conv_info(attributes.stride().x(), attributes.stride().y(), attributes.pad().left,
                            attributes.pad().right,
                            attributes.pad().top, attributes.pad().bottom, DimensionRoundingType::FLOOR);
    // Initialize the direct convolution descriptor
    const DirectConvComputeKernelInfo desc = config_direct_convolution_nhwc(src, wei, conv_info);

    ITensorInfo *dst = sketch.implementation().create_virtual_tensor();

    // Assert validation
    ARM_COMPUTE_ERROR_THROW_ON(GpuConv2d::validate_op(sketch, src, wei, bia, attributes));
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, wei, dst);

    // Auto initialize dst tensor
    calculate_and_init_dst_if_empty(dst, src, wei, attributes);

    // Translate into components and add to component graph
    auto &comp_graph = sketch.implementation().component_graph();

    const auto sketch_ctx = sketch.implementation().context();

    const auto data_layout = src->data_layout();
    const auto gpu_target  = sketch_ctx->gpu_target();

    if(sketch_ctx->gpu_language() == GpuLanguage::OpenCL)
    {
        const auto cl_compile_ctx = sketch_ctx->cl_compile_context();
        ARM_COMPUTE_ERROR_ON(cl_compile_ctx == nullptr);

        // Add Direct Conv2d Component
        {
            auto properties = IGpuKernelComponent::Properties();
            properties.stage(UnitWorkloadStage{ UnitWorkloadStage::Stage::Run });

            auto settings = ClComponentDirectConv2d::Settings();

            settings.export_to_cl_image(
                export_to_cl_image_support(src, gpu_target, cl_compile_ctx->get_device(), data_layout));

            settings.fast_relaxed_math(
                (gpu_target != GPUTarget::G71 && (gpu_target & GPUTarget::GPU_ARCH_MASK) == GPUTarget::BIFROST)
                && (dst->data_type() == DataType::F32 || dst->data_type() == DataType::F16));

            if(settings.export_to_cl_image())
            {
                arm_compute::opencl::kernels::gemm::update_padding_for_cl_image(wei);
            }

            settings.direct_conv_descriptor(desc);

            ArgumentPack<ITensorInfo> arguments;
            arguments.add_const_tensor(ACL_SRC_0, src);
            arguments.add_const_tensor(ACL_SRC_1, wei);
            arguments.add_const_tensor(ACL_SRC_2, bia);
            arguments.add_const_tensor(ACL_DST_0, dst);
            comp_graph.add_new_component<ClComponentDirectConv2d>(properties, arguments, attributes, settings);
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

    const auto op = sketch.implementation().operator_group().new_operator(operator_type, tensors);
    sketch.implementation().operator_group().add_operator(op);

    return dst;
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
