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
#ifdef ACL_INTERNAL_TEST_CKW_IN_DF

#include "src/dynamic_fusion/sketch/gpu/components/cl/ClComponentMatMul.h"

#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/dynamic_fusion/sketch/attributes/MatMulAttributes.h"

#include "src/core/CL/CLValidate.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/GpuCkwMatMul.h"
#include "src/gpu/cl/kernels/helpers/MatMulKernelHelpers.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
namespace
{
using Attributes = MatMulAttributes;
using Settings   = GpuMatMulSettings;

Status validate_matmul_kernel_info(Attributes attributes, Settings settings)
{
    const bool adj_lhs = attributes.adj_lhs();
    const bool adj_rhs = attributes.adj_rhs();
    const int  m0      = settings.m0();
    const int  n0      = settings.n0();
    const int  k0      = settings.k0();

    // Validate M0
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(m0 < 1, "Only positive integers are supported for M0");

    if (adj_lhs)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(((m0 & (m0 - 1)) && (m0 != 3)) || (m0 > 16),
                                        "Only 1,2,3,4,8,16 are supported for M0 for Lhs transposed");
    }

    // Validate N0
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(n0 < 1, "Only positive integers are supported for N0");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(((n0 & (n0 - 1)) && (n0 != 3)) || (n0 > 16),
                                    "Only 1,2,3,4,8,16 are supported for N0");

    // Validate K0
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(k0 < 1, "Only positive integers are supported for K0");
    if (!adj_lhs || adj_rhs)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(((k0 & (k0 - 1)) && (k0 != 3)) || (k0 > 16),
                                        "Only 1,2,3,4,8,16 are supported for K0");
    }

    return Status{};
}

} // namespace

Status ClComponentMatMul::validate(const Properties                &properties,
                                   const ArgumentPack<ITensorInfo> &tensors,
                                   const Attributes                &attributes,
                                   const Settings                  &settings)
{
    ARM_COMPUTE_UNUSED(properties);
    ARM_COMPUTE_UNUSED(attributes);

    const auto lhs = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const auto rhs = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    const auto dst = tensors.get_const_tensor(TensorType::ACL_DST_0);

    // Currently, the only supported case is when adj_lhs = false and adj_rhs = true
    ARM_COMPUTE_RETURN_ERROR_ON((attributes.adj_lhs() != false) && (attributes.adj_rhs() != true));

    // Check if Matching data type
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(lhs, rhs);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(lhs, dst);

    // Data type
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(lhs, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(lhs, dst);

    // All tensor infos are initialized
    ARM_COMPUTE_RETURN_ERROR_ON(lhs->tensor_shape().total_size() == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(rhs->tensor_shape().total_size() == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(dst->tensor_shape().total_size() == 0);

    // Device requirements are met
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(lhs);

    // Check if block sizes are supported
    MatMulKernelInfo matmul_kernel_info =
        MatMulKernelInfo(attributes.adj_lhs(), attributes.adj_rhs(), settings.m0(), settings.n0(), settings.k0());
    ARM_COMPUTE_RETURN_ON_ERROR(validate_matmul_kernel_info(attributes, settings));
    ARM_COMPUTE_RETURN_ON_ERROR(
        opencl::kernels::validate_matmul_input_shapes(lhs->tensor_shape(), rhs->tensor_shape(), matmul_kernel_info));

    // Check if dst shape is correct
    const auto expected_dst_shape =
        misc::shape_calculator::compute_matmul_shape(lhs->tensor_shape(), rhs->tensor_shape(), matmul_kernel_info);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(dst->tensor_shape(), expected_dst_shape);

    return Status{};
}

ClComponentMatMul::ClComponentMatMul(ComponentId                      id,
                                     const Properties                &properties,
                                     const ArgumentPack<ITensorInfo> &tensors,
                                     const Attributes                &attributes,
                                     const Settings                  &settings)
    : IGpuKernelComponent{id, properties, tensors},
      _component_writer{std::make_unique<GpuCkwMatMul>(id, tensors, attributes, settings)}
{
}

ClComponentMatMul::~ClComponentMatMul()
{
}

const IGpuCkwComponentDriver *ClComponentMatMul::ckw_component_driver() const
{
    return _component_writer.get();
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif // ACL_INTERNAL_TEST_CKW_IN_DF
