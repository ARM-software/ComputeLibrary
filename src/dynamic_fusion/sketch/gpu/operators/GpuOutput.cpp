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

#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuOutput.h"

#include "src/core/helpers/AutoConfiguration.h"
#include "src/common/utils/Log.h"

#include "src/dynamic_fusion/sketch/ArgumentPack.h"
#include "src/dynamic_fusion/sketch/gpu/GpuWorkloadSketchImpl.h"
#include "src/dynamic_fusion/sketch/gpu/components/cl/ClComponentStore.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
namespace
{
constexpr GpuOperatorType operator_type = GpuOperatorType::Simple;
}

Status GpuOutput::is_supported_op(const GpuWorkloadContext &context,
                                  const ITensorInfo        *src,
                                  const ITensorInfo        *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);

    // Initialize the destination tensor info.
    TensorInfo dst_to_validate = *dst;
    auto_init_if_empty(dst_to_validate, *src);

    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, &dst_to_validate);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, &dst_to_validate);

    ARM_COMPUTE_UNUSED(context);
    return Status{};
}

Status GpuOutput::validate_op(const GpuWorkloadSketch &sketch,
                              const ITensorInfo       *src,
                              const ITensorInfo       *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON(!src->has_valid_id());
    ARM_COMPUTE_RETURN_ERROR_ON(!dst->has_valid_id());

    // Initialize the destination tensor info.
    TensorInfo dst_to_validate = *dst;
    auto_init_if_empty(dst_to_validate, *src);

    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, &dst_to_validate);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, &dst_to_validate);

    // Perform fusion test.
    ArgumentPack<ITensorInfo> tensors;
    tensors.add_const_tensor(ACL_SRC_0, src);
    tensors.add_const_tensor(ACL_DST_0, &dst_to_validate);

    const auto group = sketch.implementation().operator_group();
    const auto op = group.new_operator(operator_type, tensors);
    const auto success = group.try_add_operator(op);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!success, "This operator cannot be fused into the workload.");
    ARM_COMPUTE_UNUSED(success);

    const auto status = is_supported_op(*sketch.gpu_context(), src, dst);
    return status;
}

void GpuOutput::create_op(GpuWorkloadSketch &sketch,
                          ITensorInfo       *src,
                          ITensorInfo       *dst)
{
    ARM_COMPUTE_LOG_PARAMS(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(GpuOutput::validate_op(sketch, src, dst));

    // Auto initialize dst tensor info if empty
    auto_init_if_empty(*dst, *src);

    // Translate into components and add to component graph
    auto &comp_graph = sketch.implementation().component_graph();
    const auto sketch_ctx = sketch.implementation().context();

    if(sketch_ctx->gpu_language() == GpuLanguage::OpenCL)
    {
        ARM_COMPUTE_ERROR_ON(sketch_ctx->cl_compile_context() == nullptr);

        // Add store component
        {
            IGpuKernelComponent::Properties properties;
            properties.stage(UnitWorkloadStage{ UnitWorkloadStage::Stage::Run });

            ArgumentPack<ITensorInfo> arguments;
            arguments.add_const_tensor(ACL_SRC_0, src);
            arguments.add_const_tensor(ACL_DST_0, dst);
            comp_graph.add_new_component<ClComponentStore>(properties, arguments);
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
