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
#include "src/gpu/cl/kernels/ClGemmMatrixMultiplyReshapedOnlyRhsMMULKernel.h"

#include "arm_compute/core/utils/ActivationFunctionUtils.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/utils/StringUtils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/CL/CLUtils.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/core/utils/helpers/float_ops.h"
#include "src/gpu/cl/kernels/gemm/ClGemmHelpers.h"
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
using ElementsProcessed = Steps;

// Block size dimensions for the MMUL extension
constexpr int mmul_m0 = 4;
constexpr int mmul_n0 = 4;
constexpr int mmul_k0 = 4;

Status validate_arguments(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, float alpha, float beta, const GEMMLHSMatrixInfo &lhs_info,
                          const GEMMRHSMatrixInfo &rhs_info,
                          const GEMMKernelInfo    &gemm_info)
{
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src0, src1, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!arm_matrix_multiply_supported(CLKernelLibrary::get().get_device()), "The extension cl_arm_matrix_multiply is not supported on the target platform");
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src0, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src0, src1);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(src0->num_dimensions() > 4, "The number of dimensions for the LHS matrix must be <= 4");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(src1->num_dimensions() > 3, "The number of dimensions for the RHS matrix must be <= 3");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(lhs_info.m0 < 1, "Only values greater than 0 are supported for m0");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(rhs_info.n0 != 1 && rhs_info.n0 != 2 && rhs_info.n0 != 3 && rhs_info.n0 != 4 && rhs_info.n0 != 8 && rhs_info.n0 != 16, "Only 1,2,3,4,8, and 16 are supported for n0");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((rhs_info.k0 != 1 || lhs_info.k0 != 1), "Only 1 is supported for k0");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((rhs_info.h0 != 4), "Only 4 is supported for h0");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(rhs_info.interleave != true, "Only true is supported for interleave with mmul extension enabled");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(rhs_info.transpose != false, "Only false is supported for transpose with mmul extension enabled");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.fp_mixed_precision, "Mixed precision not supported");
    ARM_COMPUTE_RETURN_ON_ERROR(gemm::validate_image2d_support_on_rhs(*src1, rhs_info));

    const unsigned int m = gemm_info.m;
    const unsigned int n = gemm_info.n;
    const unsigned int k = gemm_info.k;

    ARM_COMPUTE_UNUSED(m);
    ARM_COMPUTE_UNUSED(n);
    ARM_COMPUTE_UNUSED(k);

    ARM_COMPUTE_RETURN_ERROR_ON(src0->dimension(0) != k);

    // Validate the reinterpreted-as-3D-case
    if(gemm_info.depth_output_gemm3d != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(src0->dimension(1) * src0->dimension(2) != m);
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON(src0->dimension(1) != m);
    }

    // Validate the gemm-batched case
    if(src1->num_dimensions() > 2)
    {
        if(gemm_info.depth_output_gemm3d != 0)
        {
            ARM_COMPUTE_RETURN_ERROR_ON(src0->dimension(3) != src1->dimension(2));
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON(src0->dimension(2) != src1->dimension(2));
        }
    }

    if(src2 != nullptr && !(helpers::float_ops::is_zero(beta)))
    {
        const unsigned int src2_dim0 = src2->dimension(0);
        const unsigned int src2_dim1 = src2->dimension(1);

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src2, src1);
        if(gemm_info.broadcast_bias)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MSG((src2_dim1 != 1 || src2_dim0 != n), "Incorrect dimension of bias matrix which is to be broadcasted");
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MSG((src2_dim0 != n || src2_dim1 != m), "Incorrect dimension of bias matrix");
        }
    }

    TensorShape tensor_shape1{ src1->tensor_shape() };
    tensor_shape1.set(0, n);
    tensor_shape1.set(1, k);

    const TensorInfo tensor_info1          = src1->clone()->set_tensor_shape(tensor_shape1);
    const TensorInfo tensor_info_reshaped1 = src1->clone()->set_tensor_shape(misc::shape_calculator::compute_rhs_reshaped_shape(tensor_info1, rhs_info));

    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src1, &tensor_info_reshaped1);

    if(dst->total_size() != 0)
    {
        const TensorInfo tensor_info_dst = dst->clone()->set_tensor_shape(misc::shape_calculator::compute_mm_shape(*src0, *src1, gemm_info));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(dst, &tensor_info_dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src0, dst);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *src0, ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst, const GEMMLHSMatrixInfo &lhs_info,
                                                        const GEMMRHSMatrixInfo &rhs_info,
                                                        const GEMMKernelInfo    &gemm_info)
{
    ARM_COMPUTE_UNUSED(src0, src1, src2);
    bool reinterpret_output_as_3d = gemm_info.depth_output_gemm3d != 0;

    // dst tensor auto initialization if not yet initialized
    auto_init_if_empty(*dst, src0->clone()->set_tensor_shape(misc::shape_calculator::compute_mm_shape(*src0, *src1, gemm_info)));

    TensorInfo tmp_info(*dst);

    if(reinterpret_output_as_3d)
    {
        // Since the dst tensor has to be reinterpreted as 3D and the execute window is based on a 2D GEMM,
        // the window needs to be constructed on the 2D collapsed version of the tensor
        TensorShape tmp_shape(dst->tensor_shape());
        tmp_shape.collapse(2U, 1U);
        tmp_info.set_tensor_shape(tmp_shape);
    }

    Window win = calculate_max_window(tmp_info, Steps(1, 1));

    // Collapse along the Z direction
    // This collapse needs to be here in order to tune the Z dimension of LWS
    const unsigned int dimension_to_collapse = std::min(static_cast<unsigned int>(dst->num_dimensions()), 2u);
    Window             collapsed             = win.collapse(win, dimension_to_collapse);

    // Reconfigure window size, one arm_matrix_multiply kernel needs 16 threads to finish.
    Window::Dimension x_dimension = collapsed.x();
    Window::Dimension y_dimension = collapsed.y();

    // Make M and N multiple of M0 and N0 respectively
    const unsigned int ceil_to_multiple_n_n0 = ceil_to_multiple(x_dimension.end(), rhs_info.n0);
    const unsigned int ceil_to_multiple_m_m0 = ceil_to_multiple(y_dimension.end(), lhs_info.m0);

    // Divide M and N by M0 and N0 respectively
    const unsigned int n_div_n0 = ceil_to_multiple_n_n0 / rhs_info.n0;
    const unsigned int m_div_m0 = ceil_to_multiple_m_m0 / lhs_info.m0;

    // Make n_div_n0 and m_div_m0 multiple of mmul_n0 and mmul_k0 respectively
    const unsigned int ceil_to_multiple_n_div_n0_mmul_n0 = ceil_to_multiple(n_div_n0, mmul_n0);
    const unsigned int ceil_to_multiple_m_div_m0_mmul_k0 = ceil_to_multiple(m_div_m0, mmul_k0);

    // Ensure x_dimension is multiple of MMUL block size (mmul_n0 * mmul_k0)
    x_dimension.set_end(ceil_to_multiple_n_div_n0_mmul_n0 * mmul_k0);
    y_dimension.set_end(ceil_to_multiple_m_div_m0_mmul_k0 / mmul_k0);

    collapsed.set(Window::DimX, x_dimension);
    collapsed.set(Window::DimY, y_dimension);

    return std::make_pair(Status{}, collapsed);
}
} // namespace

ClGemmMatrixMultiplyReshapedOnlyRhsMMULKernel::ClGemmMatrixMultiplyReshapedOnlyRhsMMULKernel()
{
    _type = CLKernelType::GEMM;
}

void ClGemmMatrixMultiplyReshapedOnlyRhsMMULKernel::configure(const CLCompileContext &compile_context, ITensorInfo *src0, ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst, float alpha,
                                                              float                    beta,
                                                              const GEMMLHSMatrixInfo &lhs_info,
                                                              const GEMMRHSMatrixInfo &rhs_info, const GEMMKernelInfo &gemm_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src0, src1, dst);

    // dst tensor auto initialization if not yet initialized
    auto_init_if_empty(*dst, src0->clone()->set_tensor_shape(misc::shape_calculator::compute_mm_shape(*src0, *src1, gemm_info)));

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src0, src1, src2, dst, alpha, beta, lhs_info, rhs_info, gemm_info));

    auto padding_info   = get_padding_info({ src0, src1, src2, dst });
    _add_bias           = src2 != nullptr;
    _export_to_cl_image = rhs_info.export_to_cl_image;

    // Configure kernel window
    auto win_config = validate_and_configure_window(src0, src1, src2, dst, lhs_info, rhs_info, gemm_info);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);

    IClKernel::configure_internal(win_config.second);

    _m = gemm_info.m;
    _n = gemm_info.n;
    _k = gemm_info.k;

    const unsigned int m0_leftover = _m % lhs_info.m0;
    const unsigned int n0_leftover = _n % rhs_info.n0;

    // Create build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(src0->data_type()));
    build_opts.add_option_if(!(helpers::float_ops::is_one(alpha)), "-DALPHA=" + float_to_string_with_full_precision(alpha));
    build_opts.add_option_if(src2 != nullptr, "-DBETA=" + float_to_string_with_full_precision(beta));
    build_opts.add_option_if(helpers::float_ops::is_one(beta), "-DUNIT_BETA");
    build_opts.add_option_if(gemm_info.broadcast_bias, "-DBROADCAST_BIAS");
    build_opts.add_option_if(src0->data_type() == DataType::F16, "-DHALF_PRECISION");
    build_opts.add_option("-DM0=" + support::cpp11::to_string(lhs_info.m0));
    build_opts.add_option("-DN0=" + support::cpp11::to_string(rhs_info.n0));
    build_opts.add_option("-DK0=" + support::cpp11::to_string(rhs_info.k0));
    build_opts.add_option("-DM0_LEFTOVER=" + support::cpp11::to_string(m0_leftover));
    build_opts.add_option("-DN0_LEFTOVER=" + support::cpp11::to_string(n0_leftover));
    build_opts.add_option("-DMMUL_M0=" + support::cpp11::to_string(mmul_m0));
    build_opts.add_option("-DMMUL_N0=" + support::cpp11::to_string(mmul_n0));
    build_opts.add_option("-DMMUL_K0=" + support::cpp11::to_string(mmul_k0));
    build_opts.add_option("-DACTIVATION_TYPE=" + lower_string(string_from_activation_func(gemm_info.activation_info.activation())));
    build_opts.add_option("-DA_VAL=" + float_to_string_with_full_precision(gemm_info.activation_info.a()));
    build_opts.add_option("-DB_VAL=" + float_to_string_with_full_precision(gemm_info.activation_info.b()));

    std::string kernel_name("gemm_mm_reshaped_only_rhs_nt_mmul");
    kernel_name += rhs_info.export_to_cl_image ? "_texture" : "";

    // A macro guard to compile ONLY the kernel of interest
    build_opts.add_option("-D" + upper_string(kernel_name));

    // Create kernel
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += (_add_bias ? "add_bias_" : "");
    _config_id += (gemm_info.broadcast_bias ? "broadcast_bias_" : "");
    _config_id += (gemm_info.activation_info.enabled() ? "fused_activation_" : "");
    _config_id += lower_string(string_from_data_type(src0->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(_m);
    _config_id += "_";
    _config_id += support::cpp11::to_string(_n);
    _config_id += "_";
    _config_id += support::cpp11::to_string(_k);
    _config_id += "_";
    _config_id += support::cpp11::to_string(lhs_info.m0);
    _config_id += "_";
    _config_id += support::cpp11::to_string(rhs_info.n0);

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status ClGemmMatrixMultiplyReshapedOnlyRhsMMULKernel::validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, float alpha, float beta,
                                                               const GEMMLHSMatrixInfo &lhs_info,
                                                               const GEMMRHSMatrixInfo &rhs_info, const GEMMKernelInfo &gemm_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src0, src1, src2, dst, alpha, beta, lhs_info, rhs_info, gemm_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(src0->clone().get(),
                                                              src1->clone().get(),
                                                              src2 != nullptr ? src2->clone().get() : nullptr,
                                                              dst->clone().get(),
                                                              lhs_info,
                                                              rhs_info,
                                                              gemm_info)
                                .first);

    return Status{};
}

void ClGemmMatrixMultiplyReshapedOnlyRhsMMULKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const auto src0 = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_0));
    const auto src1 = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_1));
    const auto src2 = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_2));
    auto       dst  = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    ARM_COMPUTE_ERROR_ON_NULLPTR(src0, src1, dst);
    ARM_COMPUTE_ERROR_ON(_add_bias && src2 == nullptr);

    if(src1->info()->num_dimensions() < 3)
    {
        // The stride_z for matrix B must be zero if we do not slice
        ARM_COMPUTE_ERROR_ON(src1->info()->strides_in_bytes()[3] != 0);
    }

    cl::Image2D src1_image2d;

    if(_export_to_cl_image)
    {
        const TensorShape shape2d(src1->info()->dimension(0) / 4, src1->info()->dimension(1) * src1->info()->dimension(2));
        const size_t      image_row_pitch = src1->info()->strides_in_bytes()[1];

        src1_image2d = create_image2d_from_buffer(CLKernelLibrary::get().context(), src1->cl_buffer(), shape2d, src1->info()->data_type(), image_row_pitch, CLImage2DType::ReadOnly);
    }

    Window slice = window.first_slice_window_3D();

    do
    {
        unsigned int idx = 0;

        add_3d_tensor_nhw_argument(idx, src0);
        if(_export_to_cl_image)
        {
            _kernel.setArg(idx++, src1_image2d);
        }
        add_3d_tensor_nhw_argument(idx, src1);

        // Bias buffer (_add_bias == true)
        if(_add_bias)
        {
            add_3d_tensor_nhw_argument(idx, src2);
        }
        // dst buffer
        add_3d_tensor_nhw_argument(idx, dst);

        // Pass m, n and k at runtime as signed ints, to ensure results of any subtractions they could be operand in, would still be signed.
        _kernel.setArg<cl_int>(idx++, _m);
        _kernel.setArg<cl_int>(idx++, _n);
        _kernel.setArg<cl_int>(idx++, _k);

        // LWS_x should be multiple of 16 at least. (32, 2) has been chosen to have more work-items on a single core
        // LWS also enforces the order of execution of the workitems which improves cache utilization
        enqueue(queue, *this, slice, cl::NDRange(32, 2), false);
    }
    while(window.slide_window_slice_3D(slice));
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
