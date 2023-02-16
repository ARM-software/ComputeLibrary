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
#include "src/dynamic_fusion/sketch/gpu/operators/internal/GpuElementwiseBinaryCommon.h"
#include "src/common/utils/Log.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/dynamic_fusion/sketch/ArgumentPack.h"
#include "src/dynamic_fusion/sketch/gpu/GpuWorkloadSketchImpl.h"
#include "src/dynamic_fusion/sketch/gpu/components/cl/ClComponentElementwiseBinary.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
namespace
{
void calculate_and_init_dst_if_empty(ITensorInfo *dst, const ITensorInfo *lhs, const ITensorInfo *rhs)
{
    if(dst->total_size() == 0U)
    {
        const std::pair<TensorShape, ValidRegion> broadcast_pair = ITensorInfo::broadcast_shape_and_valid_region(*lhs, *rhs);
        auto_init_if_empty(*dst, lhs->clone()->set_tensor_shape(broadcast_pair.first));
    }
}

Status is_supported_op_helper(const GpuWorkloadContext                &context,
                              const ITensorInfo                       *lhs,
                              const ITensorInfo                       *rhs,
                              const ITensorInfo                       *dst,
                              const ElementwiseBinaryCommonAttributes &attributes)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(lhs, rhs);

    TensorInfo         dst_info_to_validate;
    const ITensorInfo *dst_info_to_validate_ptr = &dst_info_to_validate;

    if(dst != nullptr)
    {
        dst_info_to_validate_ptr = dst;
    }

    calculate_and_init_dst_if_empty(&dst_info_to_validate, lhs, rhs);

    // Check components
    if(context.gpu_language() == GpuLanguage::OpenCL)
    {
        const auto cl_compile_ctx = context.cl_compile_context();
        ARM_COMPUTE_RETURN_ERROR_ON(cl_compile_ctx == nullptr);

        // Validate ElementwiseBinary Component
        {
            ArgumentPack<ITensorInfo> arguments;
            arguments.add_const_tensor(ACL_SRC_0, lhs);
            arguments.add_const_tensor(ACL_SRC_1, rhs);
            arguments.add_const_tensor(ACL_DST_0, dst_info_to_validate_ptr);

            ARM_COMPUTE_RETURN_ON_ERROR(ClComponentElementwiseBinary::validate(arguments, attributes));
        }
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_MSG("Unimplemented Gpu language");
    }

    return Status{};
}

GpuOperatorType operator_type = GpuOperatorType::Simple;
} // namespace

ElementwiseBinaryCommonAttributes &ElementwiseBinaryCommonAttributes::operation(const ElementwiseBinaryCommonAttributes::ElementwiseOp &operation)
{
    _operation = operation;
    return *this;
}

ElementwiseBinaryCommonAttributes::ElementwiseOp ElementwiseBinaryCommonAttributes::operation() const
{
    return _operation;
}

Status GpuElementwiseBinaryCommon::is_supported_op(const GpuWorkloadContext                &context,
                                                   const ITensorInfo                       *lhs,
                                                   const ITensorInfo                       *rhs,
                                                   const ElementwiseBinaryCommonAttributes &attributes)
{
    return is_supported_op_helper(context, lhs, rhs, nullptr, attributes);
}

Status GpuElementwiseBinaryCommon::validate_op(const GpuWorkloadSketch                 &sketch,
                                               const ITensorInfo                       *lhs,
                                               const ITensorInfo                       *rhs,
                                               const ElementwiseBinaryCommonAttributes &attributes)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(lhs, rhs);
    ARM_COMPUTE_RETURN_ERROR_ON(!lhs->has_valid_id() || !rhs->has_valid_id());

    // Refer to GpuConv2d::validate_op() for id-validness of this TensorInfo object
    TensorInfo dst_info_to_validate;

    // Auto initialize dst tensor info
    calculate_and_init_dst_if_empty(&dst_info_to_validate, lhs, rhs);

    // Perform fusion test
    // Pack tensor infos
    ArgumentPack<ITensorInfo> tensors;
    tensors.add_const_tensor(ACL_SRC_0, lhs);
    tensors.add_const_tensor(ACL_SRC_1, rhs);
    tensors.add_const_tensor(ACL_DST_0, &dst_info_to_validate);
    const auto op = sketch.implementation().operator_group().new_operator(operator_type, tensors);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!sketch.implementation().operator_group().try_add_operator(op),
                                    "Operator fusion test failed. This operator cannot be fused into the workload");

    // Check if configuration is supported
    return is_supported_op_helper(*sketch.gpu_context(), lhs, rhs, &dst_info_to_validate, attributes);
}

ITensorInfo *GpuElementwiseBinaryCommon::create_op(GpuWorkloadSketch                       &sketch,
                                                   ITensorInfo                             *lhs,
                                                   ITensorInfo                             *rhs,
                                                   const ElementwiseBinaryCommonAttributes &attributes)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(lhs, rhs);
    ARM_COMPUTE_LOG_PARAMS(lhs, rhs);
    ARM_COMPUTE_ERROR_THROW_ON(GpuElementwiseBinaryCommon::validate_op(sketch, lhs, rhs, attributes));

    ITensorInfo *dst = sketch.implementation().create_virtual_tensor();
    ARM_COMPUTE_ERROR_ON_NULLPTR(dst);

    // Auto initialize dst tensor
    calculate_and_init_dst_if_empty(dst, lhs, rhs);

    // Translate into components and add to component graph
    auto &comp_graph = sketch.implementation().component_graph();

    const auto sketch_ctx = sketch.implementation().context();

    if(sketch_ctx->gpu_language() == GpuLanguage::OpenCL)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(sketch_ctx->cl_compile_context());

        // Add ElementwiseBinary Component
        {
            auto properties = IGpuKernelComponent::Properties();
            properties.stage(UnitWorkloadStage{ UnitWorkloadStage::Stage::Run });

            ArgumentPack<ITensorInfo> arguments;
            arguments.add_const_tensor(ACL_SRC_0, lhs);
            arguments.add_const_tensor(ACL_SRC_1, rhs);
            arguments.add_const_tensor(ACL_DST_0, dst);
            comp_graph.add_new_component<ClComponentElementwiseBinary>(properties, arguments, attributes);
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
    tensors.add_const_tensor(ACL_SRC_0, lhs);
    tensors.add_const_tensor(ACL_SRC_1, rhs);
    tensors.add_tensor(ACL_DST_0, dst);
    const auto op = sketch.implementation().operator_group().new_operator(operator_type, tensors);
    sketch.implementation().operator_group().add_operator(op);

    return dst;
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
