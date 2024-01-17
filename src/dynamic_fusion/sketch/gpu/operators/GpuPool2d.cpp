/*
 * Copyright (c) 2023-2024 Arm Limited.
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

#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuPool2d.h"

#include "arm_compute/core/CL/CLCompileContext.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/experimental/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadContext.h"
#include "arm_compute/dynamic_fusion/sketch/gpu/GpuWorkloadSketch.h"

#include "src/common/utils/Log.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/dynamic_fusion/sketch/ArgumentPack.h"
#include "src/dynamic_fusion/sketch/gpu/components/cl/ClComponentPool2d.h"
#include "src/dynamic_fusion/sketch/gpu/GpuWorkloadSketchImpl.h"
#include "src/dynamic_fusion/sketch/gpu/GpuWorkloadSourceCode.h"
#include "src/dynamic_fusion/utils/Utils.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
namespace
{
void calculate_and_init_dst_if_empty(ITensorInfo             *dst,
                                     const ITensorInfo       *src,
                                     const Pool2dAttributes  &attributes,
                                     const GpuPool2dSettings &settings)
{
    ARM_COMPUTE_UNUSED(settings);

    if (dst->total_size() == 0U)
    {
        auto shape = misc::shape_calculator::compute_pool_shape(
            *src, convert_pool_attr_to_pool_info(attributes, /* mixed_precision */ true));
        auto_init_if_empty(*dst, src->clone()->set_tensor_shape(shape));
    }
}

constexpr GpuOperatorType operator_type = GpuOperatorType::Complex;
} // namespace

GpuPool2dSettings GpuPool2dSettings::use_inf_as_limit(bool use_inf_as_limit)
{
    _use_inf_as_limit = use_inf_as_limit;
    return *this;
}

bool GpuPool2dSettings::use_inf_as_limit() const
{
    return _use_inf_as_limit;
}

Status GpuPool2d::validate_op(const GpuWorkloadSketch &sketch,
                              const ITensorInfo       *src,
                              const Pool2dAttributes  &attributes,
                              const GpuPool2dSettings &settings)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src);
    ARM_COMPUTE_RETURN_ERROR_ON(!src->has_valid_id());

    // Auto initialize dst tensor info
    TensorInfo dst_info_to_validate;

    calculate_and_init_dst_if_empty(&dst_info_to_validate, src, attributes, settings);

    // Perform fusion test
    // Pack tensor infos
    ArgumentPack<ITensorInfo> tensors;
    tensors.add_const_tensor(ACL_SRC_0, src);
    tensors.add_const_tensor(ACL_DST_0, &dst_info_to_validate);

    const auto op = sketch.implementation().operator_group().new_operator(operator_type, tensors);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!sketch.implementation().operator_group().try_add_operator(op),
                                    "Operator fusion test failed. This operator cannot be fused into the workload");

    // Check if configuration is supported
    return is_supported_op(*sketch.gpu_context(), src, attributes, settings);
}

Status GpuPool2d::is_supported_op(const GpuWorkloadContext &context,
                                  const ITensorInfo        *src,
                                  const Pool2dAttributes   &attributes,
                                  const GpuPool2dSettings  &settings)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src);
    // Data type
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::F16, DataType::F32);
    // Data layout
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(src, DataLayout::NHWC);
    // Check exclude padding is not false
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!attributes.exclude_padding(),
                                    "Exclude padding must be set to true in Attributes!");

    // Auto initialize dst tensor info
    TensorInfo dst_info_to_validate;

    calculate_and_init_dst_if_empty(&dst_info_to_validate, src, attributes, settings);

    // Check components
    if (context.gpu_language() == GpuLanguage::OpenCL)
    {
        const auto cl_compile_ctx = context.cl_compile_context();
        ARM_COMPUTE_RETURN_ERROR_ON(cl_compile_ctx == nullptr);

        // Validate Component
        {
            const KernelProperties properties =
                IGpuKernelComponent::Properties().stage(UnitWorkloadStage{UnitWorkloadStage::Stage::Run});

            ArgumentPack<ITensorInfo> arguments;
            arguments.add_const_tensor(ACL_SRC_0, src);
            arguments.add_const_tensor(ACL_DST_0, &dst_info_to_validate);
            ARM_COMPUTE_RETURN_ON_ERROR(ClComponentPool2d::validate(properties, arguments, attributes, settings));
        }
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_MSG("Unimplemented Gpu language");
    }
    return Status{};
}

ITensorInfo *GpuPool2d::create_op(GpuWorkloadSketch       &sketch,
                                  ITensorInfo             *src,
                                  const Pool2dAttributes  &attributes,
                                  const GpuPool2dSettings &settings)
{
    // Assert validation
    ARM_COMPUTE_ERROR_THROW_ON(GpuPool2d::validate_op(sketch, src, attributes, settings));
    ARM_COMPUTE_LOG_PARAMS(src, attributes, settings);

    ITensorInfo *dst = sketch.implementation().create_virtual_tensor();
    ARM_COMPUTE_ERROR_ON_NULLPTR(dst);

    // Auto initialize dst tensor
    calculate_and_init_dst_if_empty(dst, src, attributes, settings);

    // Translate into components and add to component graph
    auto &comp_graph = sketch.implementation().component_graph();

    const auto sketch_ctx = sketch.implementation().context();

    if (sketch_ctx->gpu_language() == GpuLanguage::OpenCL)
    {
        const auto cl_compile_ctx = sketch_ctx->cl_compile_context();
        ARM_COMPUTE_UNUSED(cl_compile_ctx);
        ARM_COMPUTE_ERROR_ON(cl_compile_ctx == nullptr);

        // Add Component
        {
            auto properties = IGpuKernelComponent::Properties();
            properties.stage(UnitWorkloadStage{UnitWorkloadStage::Stage::Run});

            ArgumentPack<ITensorInfo> arguments;
            arguments.add_const_tensor(ACL_SRC_0, src);
            arguments.add_const_tensor(ACL_DST_0, dst);
            comp_graph.add_new_component<ClComponentPool2d>(properties, arguments, attributes, settings);
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
    tensors.add_tensor(ACL_DST_0, dst);

    const auto op = sketch.implementation().operator_group().new_operator(operator_type, tensors);
    sketch.implementation().operator_group().add_operator(op);

    return dst;
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
