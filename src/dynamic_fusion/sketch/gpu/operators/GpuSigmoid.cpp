/*
 * Copyright (c) 2023 Arm Limited.
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

#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuSigmoid.h"
#include "arm_compute/core/ActivationLayerInfo.h"
#include "arm_compute/core/experimental/Types.h"

#include "src/common/utils/Log.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/dynamic_fusion/sketch/ArgumentPack.h"
#include "src/dynamic_fusion/sketch/gpu/GpuWorkloadSketchImpl.h"
#include "src/dynamic_fusion/sketch/gpu/components/cl/ClComponentActivation.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
namespace
{
Status is_supported_op_helper(const GpuWorkloadContext &context,
                              const ITensorInfo        *src,
                              const ITensorInfo        *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::F16, DataType::F32);

    TensorInfo         dst_info_to_validate;
    const ITensorInfo *dst_info_to_validate_ptr = &dst_info_to_validate;

    if(dst != nullptr)
    {
        dst_info_to_validate_ptr = dst;
    }

    auto_init_if_empty(dst_info_to_validate, *src->clone());

    const ClComponentActivation::Attributes act_info{ ActivationLayerInfo::ActivationFunction::LOGISTIC };

    // Check components
    if(context.gpu_language() == GpuLanguage::OpenCL)
    {
        // Validate Activation Component
        const auto properties = IGpuKernelComponent::Properties().stage(UnitWorkloadStage{ UnitWorkloadStage::Stage::Run });

        ArgumentPack<ITensorInfo> arguments;
        arguments.add_const_tensor(ACL_SRC, src);
        arguments.add_const_tensor(ACL_DST, dst_info_to_validate_ptr);
        ARM_COMPUTE_RETURN_ON_ERROR(ClComponentActivation::validate(properties, arguments, act_info));
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_MSG("Unimplemented Gpu language");
    }
    return Status{};
}

constexpr GpuOperatorType operator_type = GpuOperatorType::Simple;
} // namespace

Status GpuSigmoid::is_supported_op(const GpuWorkloadContext &context,
                                   const ITensorInfo        *src)
{
    return is_supported_op_helper(context, src, nullptr);
}

Status GpuSigmoid::validate_op(const GpuWorkloadSketch &sketch,
                               const ITensorInfo       *src)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src);

    // Check if tensors have valid id, i.e. they are created from a sketch
    ARM_COMPUTE_RETURN_ERROR_ON(!src->has_valid_id());

    // Refer to GpuConv2d::validate_op() for id-validness of this TensorInfo object
    TensorInfo dst_info_to_validate;

    // Auto initialize dst tensor info
    auto_init_if_empty(dst_info_to_validate, *src->clone());

    // Perform fusion test to check if the operator meets fusion constraints
    ArgumentPack<ITensorInfo> tensors;
    tensors.add_const_tensor(ACL_SRC, src);
    tensors.add_const_tensor(ACL_DST, &dst_info_to_validate);
    const auto op = sketch.implementation().operator_group().new_operator(operator_type, tensors);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!sketch.implementation().operator_group().try_add_operator(op),
                                    "Operator fusion test failed. This operator cannot be fused into the workload");

    // Check if configuration is supported
    return is_supported_op_helper(*sketch.gpu_context(), src, &dst_info_to_validate);
}

ITensorInfo *GpuSigmoid::create_op(GpuWorkloadSketch &sketch,
                                   ITensorInfo       *src)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src);
    ARM_COMPUTE_LOG_PARAMS(src);
    ARM_COMPUTE_ERROR_THROW_ON(GpuSigmoid::validate_op(sketch, src));

    ITensorInfo *dst = sketch.implementation().create_virtual_tensor();
    ARM_COMPUTE_ERROR_ON_NULLPTR(dst);

    // Auto initialize dst tensor
    auto_init_if_empty(*dst, *src->clone());

    // Translate into components and add to component graph
    GpuKernelComponentGraph &comp_graph = sketch.implementation().component_graph();

    const ClComponentActivation::Attributes act_info{ ActivationLayerInfo::ActivationFunction::LOGISTIC };

    const auto *const sketch_ctx = sketch.implementation().context();

    if(sketch_ctx->gpu_language() == GpuLanguage::OpenCL)
    {
        // Add Activation Component
        auto properties = IGpuKernelComponent::Properties();
        properties.stage(UnitWorkloadStage{ UnitWorkloadStage::Stage::Run });

        ArgumentPack<ITensorInfo> arguments;
        arguments.add_const_tensor(ACL_SRC, src);
        arguments.add_const_tensor(ACL_DST, dst);
        comp_graph.add_new_component<ClComponentActivation>(properties, arguments, act_info);
    }
    else
    {
        ARM_COMPUTE_ERROR("Unimplemented Gpu language");
    }

    // Set up fusion test by adding to the Operator Group
    // Note this has to be performed after all the components have been successfully added to the component graph

    // Pack tensor infos
    ArgumentPack<ITensorInfo> tensors;
    tensors.add_const_tensor(ACL_SRC, src);
    tensors.add_const_tensor(ACL_DST, dst);

    const auto op = sketch.implementation().operator_group().new_operator(operator_type, tensors);
    sketch.implementation().operator_group().add_operator(op);

    return dst;
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
