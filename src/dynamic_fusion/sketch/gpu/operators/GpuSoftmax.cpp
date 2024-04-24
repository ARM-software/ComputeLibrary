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
#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuSoftmax.h"

#include "arm_compute/core/Error.h"

#include "src/common/utils/Log.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/dynamic_fusion/sketch/ArgumentPack.h"
#include "src/dynamic_fusion/sketch/gpu/components/cl/ClComponentLogits1DMaxShiftExpSum.h"
#include "src/dynamic_fusion/sketch/gpu/components/cl/ClComponentLogits1DNorm.h"
#include "src/dynamic_fusion/sketch/gpu/GpuOperatorProperties.h"
#include "src/dynamic_fusion/sketch/gpu/GpuWorkloadSketchImpl.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
namespace
{
GpuOperatorType operator_type = GpuOperatorType::Unfusable;
} // namespace

Status GpuSoftmax::is_supported_op(const GpuWorkloadContext &context,
                                   const ITensorInfo        *src,
                                   const ITensorInfo        *dst,
                                   const Attributes         &attributes)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    TensorInfo dst_info_to_validate;

    // Auto initialize dst tensor info
    if (dst != nullptr)
    {
        dst_info_to_validate = *dst;
    }
    else
    {
        auto_init_if_empty(dst_info_to_validate, *src->clone());
    }
    // Check components
    if (context.gpu_language() == GpuLanguage::OpenCL)
    {
        const auto cl_compile_ctx = context.cl_compile_context();
        ARM_COMPUTE_RETURN_ERROR_ON(cl_compile_ctx == nullptr);
        const KernelProperties properties =
            IGpuKernelComponent::Properties().stage(UnitWorkloadStage{UnitWorkloadStage::Stage::Run});

        TensorShape logits_sum_shape = src->tensor_shape();
        TensorInfo  logits(src->clone()->set_tensor_shape(logits_sum_shape));

        // The sum tensor dim0 only need one element
        logits_sum_shape.set(0, 1);
        TensorInfo sum(src->clone()->set_tensor_shape(logits_sum_shape));

        // Validate Component
        ArgumentPack<ITensorInfo> arguments_exp_sum;
        ArgumentPack<ITensorInfo> arguments_norm;

        arguments_exp_sum.add_const_tensor(ACL_SRC_0, src);
        arguments_exp_sum.add_const_tensor(ACL_DST_0, &sum);
        arguments_exp_sum.add_const_tensor(ACL_DST_1, &logits);

        arguments_norm.add_const_tensor(ACL_SRC_0, &logits);
        arguments_norm.add_const_tensor(ACL_SRC_1, &sum);
        arguments_norm.add_const_tensor(ACL_DST_0, &dst_info_to_validate);

        ARM_COMPUTE_RETURN_ON_ERROR(
            ClComponentLogits1DMaxShiftExpSum::validate(properties, arguments_exp_sum, attributes));
        ARM_COMPUTE_RETURN_ON_ERROR(ClComponentLogits1DNorm::validate(properties, arguments_norm, attributes));
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_MSG("Unimplemented Gpu language");
    }

    return Status{};
}

Status GpuSoftmax::validate_op(const GpuWorkloadSketch &sketch,
                               const ITensorInfo       *src,
                               const ITensorInfo       *dst,
                               const Attributes        &attributes)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON(!src->has_valid_id() || !dst->has_valid_id());
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(src->num_dimensions() > 4, "Only up to 4 dimensions are supported");
    ARM_COMPUTE_RETURN_ERROR_ON(attributes.axis() < static_cast<int32_t>(-src->num_dimensions()) ||
                                static_cast<int32_t>(src->num_dimensions()) <= attributes.axis());

    // Auto initialize dst tensor info
    TensorInfo dst_info_to_validate = *dst;
    auto_init_if_empty(dst_info_to_validate, *src->clone());

    const size_t actual_axis =
        static_cast<size_t>(wrap_around(attributes.axis(), static_cast<int32_t>(src->num_dimensions())));
    const bool needs_permute = actual_axis != 0;
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(needs_permute, "Dynamic fusion softmax on axis!=0 not supported yet.");

    // Perform fusion test and check if the operator meets the fusion constraints
    ArgumentPack<ITensorInfo> tensors;
    tensors.add_const_tensor(ACL_SRC_0, src);
    tensors.add_const_tensor(ACL_DST_0, &dst_info_to_validate);

    const auto op = sketch.implementation().operator_group().new_operator(operator_type, tensors);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!sketch.implementation().operator_group().try_add_operator(op),
                                    "Operator fusion test failed. This operator cannot be fused into the workload");

    // Check if configuration is supported
    return is_supported_op(*sketch.gpu_context(), src, &dst_info_to_validate, attributes);
}

void GpuSoftmax::create_op(GpuWorkloadSketch &sketch, ITensorInfo *src, ITensorInfo *dst, const Attributes &attributes)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_LOG_PARAMS(src, dst, attributes);
    TensorShape  logits_sum_shape = src->tensor_shape();
    ITensorInfo *logits           = sketch.implementation().create_auxiliary_tensor(
                  src->clone()->set_id(ITensorInfo::invalid_tensor_id).set_tensor_shape(logits_sum_shape));
    logits_sum_shape.set(0, 1);
    ITensorInfo *sum = sketch.implementation().create_auxiliary_tensor(
        src->clone()->set_id(ITensorInfo::invalid_tensor_id).set_tensor_shape(logits_sum_shape));

    // Auto initialize dst tensor info and the auxiliary tensor infos as well
    auto_init_if_empty(*dst, *src->clone());

    // Assert validation
    ARM_COMPUTE_ERROR_THROW_ON(GpuSoftmax::validate_op(sketch, src, dst, attributes));
    ARM_COMPUTE_ERROR_ON_NULLPTR(logits, sum);

    // Translate into components and add to component graph
    auto      &comp_graph = sketch.implementation().component_graph();
    const auto sketch_ctx = sketch.implementation().context();

    if (sketch_ctx->gpu_language() == GpuLanguage::OpenCL)
    {
        const auto cl_compile_ctx = sketch_ctx->cl_compile_context();
        ARM_COMPUTE_UNUSED(cl_compile_ctx);
        ARM_COMPUTE_ERROR_ON(cl_compile_ctx == nullptr);

        // Add Direct Conv2d Component
        {
            auto properties = IGpuKernelComponent::Properties();
            properties.stage(UnitWorkloadStage{UnitWorkloadStage::Stage::Run});

            ArgumentPack<ITensorInfo> arguments_exp_sum;
            ArgumentPack<ITensorInfo> arguments_norm;

            arguments_exp_sum.add_const_tensor(ACL_SRC_0, src);
            arguments_exp_sum.add_const_tensor(ACL_DST_0, sum);
            arguments_exp_sum.add_const_tensor(ACL_DST_1, logits);

            arguments_norm.add_const_tensor(ACL_SRC_0, logits);
            arguments_norm.add_const_tensor(ACL_SRC_1, sum);
            arguments_norm.add_const_tensor(ACL_DST_0, dst);

            comp_graph.add_new_component<ClComponentLogits1DMaxShiftExpSum>(properties, arguments_exp_sum, attributes);
            comp_graph.add_new_component<ClComponentLogits1DNorm>(properties, arguments_norm, attributes);
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

    const auto op = sketch.implementation().operator_group().new_operator(operator_type, tensors);
    sketch.implementation().operator_group().add_operator(op);
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
