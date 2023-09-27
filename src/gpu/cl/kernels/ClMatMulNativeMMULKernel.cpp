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
#include "src/gpu/cl/kernels/ClMatMulNativeMMULKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/utils/helpers/AdjustVecSize.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/StringUtils.h"

#include "src/common/utils/Log.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/gpu/cl/kernels/helpers/MatMulKernelHelpers.h"
#include "support/Cast.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
namespace
{
// Block size dimensions for the MMUL extension
constexpr int mmul_m0 = 4;
constexpr int mmul_n0 = 4;
constexpr int mmul_k0 = 4;

Status validate_matmul_kernel_info(const MatMulKernelInfo &matmul_kernel_info)
{
    const bool adj_lhs = matmul_kernel_info.adj_lhs;
    const int  m0      = matmul_kernel_info.m0;
    const int  n0      = matmul_kernel_info.n0;
    const int  k0      = matmul_kernel_info.k0;

    // Validate M0
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(m0 < 1, "Only positive integers are supported for M0");

    if (adj_lhs)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG((m0 != 1) && (m0 != 2) && (m0 != 3) && (m0 != 4) && (m0 != 8) && (m0 != 16),
                                        "Only 1,2,3,4,8,16 are supported for M0 for Lhs transposed");
    }

    // Validate N0
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(n0 < 1, "Only positive integers are supported for N0");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((n0 != 1) && (n0 != 2) && (n0 != 3) && (n0 != 4) && (n0 != 8) && (n0 != 16),
                                    "Only 1,2,3,4,8,16 are supported for N0");

    // Validate K0
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((k0 != 1), "Only 1 is supported for k0");

    return Status{};
}
} // namespace
ClMatMulNativeMMULKernel::ClMatMulNativeMMULKernel()
{
    _type = CLKernelType::GEMM;
}

Status ClMatMulNativeMMULKernel::validate(const ITensorInfo      *lhs,
                                          const ITensorInfo      *rhs,
                                          const ITensorInfo      *bias,
                                          const ITensorInfo      *dst,
                                          const MatMulKernelInfo &matmul_kernel_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(lhs, rhs, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(lhs, 1, DataType::F32, DataType::F16);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!arm_matrix_multiply_supported(CLKernelLibrary::get().get_device()),
                                    "The extension cl_arm_matrix_multiply is not supported on the target platform");
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(lhs, rhs);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_matmul_kernel_info(matmul_kernel_info));

    const TensorShape &lhs_shape = lhs->tensor_shape();
    ARM_COMPUTE_RETURN_ON_ERROR(validate_matmul_input_shapes(lhs_shape, rhs->tensor_shape(), matmul_kernel_info));

    const size_t lhs_k = matmul_kernel_info.adj_lhs ? lhs_shape.y() : lhs_shape.x();
    ARM_COMPUTE_RETURN_ERROR_ON_MSG_VAR((lhs_k % mmul_k0) != 0, "K dimension must be a multiple of %d", mmul_k0);

    const TensorShape expected_output_shape =
        misc::shape_calculator::compute_matmul_shape(lhs_shape, rhs->tensor_shape(), matmul_kernel_info);

    if (dst->total_size() != 0)
    {
        const TensorInfo tensor_info_dst = dst->clone()->set_tensor_shape(expected_output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(dst, &tensor_info_dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(lhs, dst);
    }

    if (bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG((bias->num_dimensions() > 1), "Multi dimensional bias is unsupported.");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(bias->dimension(0) != expected_output_shape[0],
                                        "First dimension of bias and output tensors must match.");
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(lhs, bias);
    }

    return Status{};
}
void ClMatMulNativeMMULKernel::configure(const ClCompileContext &compile_context,
                                         ITensorInfo            *lhs,
                                         ITensorInfo            *rhs,
                                         ITensorInfo            *bias,
                                         ITensorInfo            *dst,
                                         const MatMulKernelInfo &matmul_kernel_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(lhs, rhs, dst);
    ARM_COMPUTE_LOG_PARAMS(lhs, rhs, bias, dst, matmul_kernel_info);
    ARM_COMPUTE_ERROR_THROW_ON(validate(lhs, rhs, bias, dst, matmul_kernel_info));

    // dst tensor auto initialization if not yet initialized
    auto_init_if_empty(*dst, lhs->clone()->set_tensor_shape(misc::shape_calculator::compute_matmul_shape(
                                 lhs->tensor_shape(), rhs->tensor_shape(), matmul_kernel_info)));

    const int m = dst->dimension(1);
    const int n = dst->dimension(0);
    const int k = matmul_kernel_info.adj_lhs ? lhs->tensor_shape().y() : lhs->tensor_shape().x();

    _m = m;
    _n = n;
    _k = k;

    const int m0 = std::min(matmul_kernel_info.m0, m);
    const int n0 = adjust_vec_size(matmul_kernel_info.n0, n);

    // Configure kernel window
    const auto win_config =
        validate_and_configure_window_for_mmul_kernels(lhs, rhs, dst, matmul_kernel_info, mmul_m0, mmul_n0);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    IClKernel::configure_internal(win_config.second);

    // Calculate partial (store instead of load) M0 and partial N0 for the partial blocks at the end of a row/column if any. This is to avoid padding.
    const unsigned int m0_leftover = m % m0;
    const unsigned int n0_leftover = n % n0;

    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(lhs->data_type()));
    build_opts.add_option_if(lhs->data_type() == DataType::F16, "-DHALF_PRECISION");
    build_opts.add_option("-DM0=" + support::cpp11::to_string(m0));
    build_opts.add_option("-DN0=" + support::cpp11::to_string(n0));
    build_opts.add_option("-DM0_LEFTOVER=" + support::cpp11::to_string(m0_leftover));
    build_opts.add_option("-DN0_LEFTOVER=" + support::cpp11::to_string(n0_leftover));
    build_opts.add_option("-DMMUL_M0=" + support::cpp11::to_string(mmul_m0));
    build_opts.add_option("-DMMUL_N0=" + support::cpp11::to_string(mmul_n0));
    build_opts.add_option("-DMMUL_K0=" + support::cpp11::to_string(mmul_k0));
    build_opts.add_option_if(bias != nullptr, "-DBIAS");

    std::string kernel_name("mat_mul_native_mmul");
    kernel_name += matmul_kernel_info.adj_lhs ? "_t" : "_nt";
    kernel_name += matmul_kernel_info.adj_rhs ? "_t" : "_nt";

    // A macro guard to compile ONLY the kernel of interest
    build_opts.add_option("-D" + upper_string(kernel_name));

    // Create kernel
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(lhs->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(k);
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(2));
    _config_id += "_";
    _config_id += support::cpp11::to_string(m0);
    _config_id += "_";
    _config_id += support::cpp11::to_string(n0);
    _config_id += "_";
    _config_id += support::cpp11::to_string(matmul_kernel_info.k0);
}

void ClMatMulNativeMMULKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const ICLTensor *lhs =
        utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_0));
    const ICLTensor *rhs =
        utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_1));
    const ICLTensor *bias = utils::cast::polymorphic_downcast<const ICLTensor *>(
        tensors.get_const_tensor(TensorType::ACL_SRC_2)); // nullptr if bias is not present
    ICLTensor *dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));
    ARM_COMPUTE_ERROR_ON_NULLPTR(lhs, rhs, dst);
    ARM_COMPUTE_LOG_PARAMS(lhs, rhs, bias, dst);
    unsigned int idx = 0;

    add_3d_tensor_nhw_argument(idx, lhs);
    add_3d_tensor_nhw_argument(idx, rhs);
    if (bias != nullptr)
    {
        add_3d_tensor_nhw_argument(idx, bias);
    }
    add_3d_tensor_nhw_argument(idx, dst);

    // Pass m and n at runtime as signed ints, to ensure results of any subtractions they could be operand in, would still be signed.
    _kernel.setArg<cl_int>(idx++, _m);
    _kernel.setArg<cl_int>(idx++, _n);
    _kernel.setArg<cl_int>(idx++, _k);

    // LWS_x should be multiple of 16 at least. (32, 2) has been chosen to have more work-items on a single core
    // LWS also enforces the order of execution of the work items which improves cache utilization
    enqueue(queue, *this, window, cl::NDRange(32, 2), false);
}

} // namespace kernels
} // namespace opencl
} // namespace arm_compute
