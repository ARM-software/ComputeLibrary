/*
 * Copyright (c) 2022-2024 Arm Limited.
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
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuCast.h"

#include "src/common/utils/Log.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/dynamic_fusion/sketch/ArgumentPack.h"
#include "src/dynamic_fusion/sketch/gpu/components/cl/ClComponentCast.h"
#include "src/dynamic_fusion/sketch/gpu/GpuWorkloadSketchImpl.h"

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
                              const ITensorInfo        *dst,
                              const CastAttributes     &attributes)
{
    ARM_COMPUTE_RETURN_ERROR_ON(src == dst);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);

    TensorInfo         dst_info_to_validate;
    const ITensorInfo *dst_info_to_validate_ptr = &dst_info_to_validate;

    if (dst != nullptr)
    {
        dst_info_to_validate_ptr = dst;
    }

    auto_init_if_empty(dst_info_to_validate, src->clone()->set_data_type(attributes.data_type()));

    // Check support level
    // Data Type
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst_info_to_validate_ptr, 1, DataType::F16, DataType::F32);

    if (context.gpu_language() == GpuLanguage::OpenCL)
    {
        const auto cl_compile_ctx = context.cl_compile_context();
        ARM_COMPUTE_RETURN_ERROR_ON(cl_compile_ctx == nullptr);
        // Validate Cast Component
        {
            const auto properties =
                IGpuKernelComponent::Properties().stage(UnitWorkloadStage{UnitWorkloadStage::Stage::Run});
            auto settings = ClComponentCast::Settings();

            ArgumentPack<ITensorInfo> arguments;
            arguments.add_const_tensor(ACL_SRC_0, src);
            arguments.add_const_tensor(ACL_DST_0, dst_info_to_validate_ptr);
            ARM_COMPUTE_RETURN_ON_ERROR(ClComponentCast::validate(properties, arguments, attributes, settings));
        }
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_MSG("Unimplemented Gpu language");
    }

    return Status{};
}
constexpr GpuOperatorType operator_type = GpuOperatorType::Simple;
} // namespace

Status
GpuCast::is_supported_op(const GpuWorkloadContext &context, const ITensorInfo *src, const CastAttributes &attributes)
{
    return is_supported_op_helper(context, src, nullptr, attributes);
}

Status GpuCast::validate_op(const GpuWorkloadSketch &sketch, const ITensorInfo *src, const CastAttributes &attributes)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src);
    ARM_COMPUTE_RETURN_ERROR_ON(!src->has_valid_id());

    // Refer to GpuConv2d::validate_op() for id-validness of this TensorInfo object
    TensorInfo dst_info_to_validate;

    // Auto initialize dst tensor info
    auto_init_if_empty(dst_info_to_validate, src->clone()->set_data_type(attributes.data_type()));

    // Perform fusion test
    // Pack tensor infos
    ArgumentPack<ITensorInfo> tensors;
    tensors.add_const_tensor(ACL_SRC_0, src);
    tensors.add_const_tensor(ACL_DST_0, &dst_info_to_validate);
    const auto op = sketch.implementation().operator_group().new_operator(operator_type, tensors);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!sketch.implementation().operator_group().try_add_operator(op),
                                    "Operator fusion test failed. This operator cannot be fused into the workload");

    // Check if configuration is supported
    return is_supported_op_helper(*sketch.gpu_context(), src, &dst_info_to_validate, attributes);
}

ITensorInfo *GpuCast::create_op(GpuWorkloadSketch &sketch, ITensorInfo *src, const CastAttributes &attributes)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src);
    ARM_COMPUTE_LOG_PARAMS(src, attributes);
    ARM_COMPUTE_ERROR_THROW_ON(GpuCast::validate_op(sketch, src, attributes));

    ITensorInfo *dst = sketch.implementation().create_virtual_tensor();
    ARM_COMPUTE_ERROR_ON_NULLPTR(dst);

    // Auto initialize dst tensor info if empty
    auto_init_if_empty(*dst, src->clone()->set_data_type(attributes.data_type()));

    // Translate into components and add to component graph
    GpuKernelComponentGraph &comp_graph = sketch.implementation().component_graph();
    const auto              *sketch_ctx = sketch.implementation().context();

    if (sketch_ctx->gpu_language() == GpuLanguage::OpenCL)
    {
        ARM_COMPUTE_ERROR_ON(sketch_ctx->cl_compile_context() == nullptr);

        // Add Depthwise Conv2d Component
        {
            const auto properties =
                IGpuKernelComponent::Properties().stage(UnitWorkloadStage{UnitWorkloadStage::Stage::Run});
            auto settings = ClComponentCast::Settings();

            ArgumentPack<ITensorInfo> arguments;
            arguments.add_const_tensor(ACL_SRC_0, src);
            arguments.add_const_tensor(ACL_DST_0, dst);
            comp_graph.add_new_component<ClComponentCast>(properties, arguments, attributes, settings);
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
    tensors.add_const_tensor(ACL_DST_0, dst);

    const Operator op = sketch.implementation().operator_group().new_operator(operator_type, tensors);
    sketch.implementation().operator_group().add_operator(op);

    return dst;
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
