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

#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuResize.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/dynamic_fusion/sketch/ArgumentPack.h"
#include "src/dynamic_fusion/sketch/gpu/GpuWorkloadSketchImpl.h"
#include "src/dynamic_fusion/sketch/gpu/components/cl/ClComponentResize.h"

#include "src/common/utils/Log.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
namespace
{
void calculate_and_init_dst_if_empty(ITensorInfo *dst, const ITensorInfo *src, const ResizeAttributes &attributes)
{
    if(dst->total_size() == 0U)
    {
        TensorShape out_shape = src->tensor_shape();

        out_shape.set(1, attributes.output_width());
        out_shape.set(2, attributes.output_height());

        auto_init_if_empty(*dst, src->clone()->set_tensor_shape(out_shape));
    }
}

constexpr GpuOperatorType operator_type = GpuOperatorType::Complex;
}
Status GpuResize::is_supported_op(const GpuWorkloadContext &context,
                                  const ITensorInfo        *src,
                                  const ITensorInfo        *dst,
                                  const Attributes         &attributes)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);

    // Auto initialize dst tensor info
    TensorInfo dst_info_to_validate = *dst;
    calculate_and_init_dst_if_empty(&dst_info_to_validate, src, attributes);

    // Check support level
    // Data type
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::U8, DataType::S16, DataType::F16, DataType::F32);
    // Data layout
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(src, DataLayout::NHWC);
    // Interpolation policy
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(attributes.interpolation_policy() != InterpolationPolicy::NEAREST_NEIGHBOR && attributes.interpolation_policy() != InterpolationPolicy::BILINEAR,
                                    "Interpolation policy must be NEAREST_NEIGHBOR or BILINEAR");

    // Check components
    if(context.gpu_language() == GpuLanguage::OpenCL)
    {
        const auto cl_compile_ctx = context.cl_compile_context();
        ARM_COMPUTE_RETURN_ERROR_ON(cl_compile_ctx == nullptr);

        // Validate Activation Component
        {
            const KernelProperties properties = IGpuKernelComponent::Properties().stage(UnitWorkloadStage{ UnitWorkloadStage::Stage::Run });

            ArgumentPack<ITensorInfo> arguments;
            arguments.add_const_tensor(ACL_SRC_0, src);
            arguments.add_const_tensor(ACL_DST_0, &dst_info_to_validate);
            ARM_COMPUTE_RETURN_ON_ERROR(ClComponentResize::validate(properties, arguments, attributes));
        }
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_MSG("Unimplemented Gpu language");
    }

    return Status{};
}

Status GpuResize::validate_op(const GpuWorkloadSketch     &sketch,
                              const ITensorInfo           *src,
                              const ITensorInfo           *dst,
                              const GpuResize::Attributes &attributes)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON(!src->has_valid_id() || !dst->has_valid_id());

    // Auto initialize dst tensor info if empty
    TensorInfo dst_info_to_validate = *dst;
    calculate_and_init_dst_if_empty(&dst_info_to_validate, src, attributes);

    // Perform fusion test
    // Pack tensor infos
    ArgumentPack<ITensorInfo> tensors;
    tensors.add_const_tensor(ACL_SRC_0, src);
    tensors.add_const_tensor(ACL_DST_0, &dst_info_to_validate);
    const Operator op = sketch.implementation().operator_group().new_operator(operator_type, tensors);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!sketch.implementation().operator_group().try_add_operator(op),
                                    "Operator fusion test failed. This operator cannot be fused into the workload");

    // Check if configuration is supported
    return is_supported_op(*sketch.gpu_context(), src, &dst_info_to_validate, attributes);
}

void GpuResize::create_op(GpuWorkloadSketch           &sketch,
                          ITensorInfo                 *src,
                          ITensorInfo                 *dst,
                          const GpuResize::Attributes &attributes)
{
    // Assert validation
    ARM_COMPUTE_ERROR_THROW_ON(GpuResize::validate_op(sketch, src, dst, attributes));
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_LOG_PARAMS(src, dst, attributes);

    // Auto initialize dst tensor info if empty
    calculate_and_init_dst_if_empty(dst, src, attributes);

    // Translate into components and add to component graph
    GpuKernelComponentGraph &comp_graph = sketch.implementation().component_graph();
    const auto              *sketch_ctx = sketch.implementation().context();

    if(sketch_ctx->gpu_language() == GpuLanguage::OpenCL)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(sketch_ctx->cl_compile_context());

        // Add Resize Component
        {
            const auto properties = IGpuKernelComponent::Properties().stage(UnitWorkloadStage{ UnitWorkloadStage::Stage::Run });

            ArgumentPack<ITensorInfo> arguments;
            arguments.add_const_tensor(ACL_SRC_0, src);
            arguments.add_const_tensor(ACL_DST_0, dst);
            comp_graph.add_new_component<ClComponentResize>(properties, arguments, attributes);
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
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
