/*
 * Copyright (c) 2018-2021, 2023 Arm Limited.
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
#include "src/gpu/cl/kernels/ClGemmMatrixMultiplyReshapedKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/CL/CLUtils.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/experimental/PostOpUtils.h"
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

const auto post_op_utils = experimental::PostOpCLKernelUtils(
{
    //  PostOp sequence                   -> {Kernel Postfix, PostOp Slots}
    { {}, { "", {} } },
    { { experimental::PostOpType::Activation }, { "", { 1 } } },

    { { experimental::PostOpType::Eltwise_Add }, { "_post_act_eltwise_op_act", { 2 } } },
    { { experimental::PostOpType::Eltwise_PRelu }, { "_post_act_eltwise_op_act", { 2 } } },

    { { experimental::PostOpType::Activation, experimental::PostOpType::Eltwise_Add }, { "_post_act_eltwise_op_act", { 1, 2 } } },
    { { experimental::PostOpType::Activation, experimental::PostOpType::Eltwise_PRelu }, { "_post_act_eltwise_op_act", { 1, 2 } } },

    { { experimental::PostOpType::Eltwise_Add, experimental::PostOpType::Activation }, { "_post_act_eltwise_op_act", { 2, 3 } } },
    { { experimental::PostOpType::Eltwise_PRelu, experimental::PostOpType::Activation }, { "_post_act_eltwise_op_act", { 2, 3 } } },

    { { experimental::PostOpType::Activation, experimental::PostOpType::Eltwise_Add, experimental::PostOpType::Activation }, { "_post_act_eltwise_op_act", { 1, 2, 3 } } },
    { { experimental::PostOpType::Activation, experimental::PostOpType::Eltwise_PRelu, experimental::PostOpType::Activation }, { "_post_act_eltwise_op_act", { 1, 2, 3 } } }
});

Status validate_arguments(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, float alpha, float beta, const GEMMLHSMatrixInfo &lhs_info,
                          const GEMMRHSMatrixInfo &rhs_info,
                          const GEMMKernelInfo    &gemm_info)
{
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src0, src1, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src0);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src0, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src0, src1);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(src0->num_dimensions() > 4, "The number of dimensions for the LHS matrix must be <= 4");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(src1->num_dimensions() > 3, "The number of dimensions for the RHS matrix must be <= 3");
    ARM_COMPUTE_RETURN_ERROR_ON(lhs_info.k0 != rhs_info.k0);
    ARM_COMPUTE_RETURN_ERROR_ON(lhs_info.transpose == rhs_info.transpose);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(((lhs_info.k0 & (lhs_info.k0 - 1)) && lhs_info.k0 != 3), "Only 2,3,4,8,16 are supported for k0");
    ARM_COMPUTE_RETURN_ERROR_ON(lhs_info.k0 > 16);
    ARM_COMPUTE_RETURN_ERROR_ON(lhs_info.m0 < 2 || lhs_info.m0 > 8);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((lhs_info.transpose) && ((lhs_info.m0 & (lhs_info.m0 - 1)) && lhs_info.m0 != 3), "Only 2,3,4,8,16 are supported for m0");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((rhs_info.transpose) && ((rhs_info.n0 & (rhs_info.n0 - 1)) && rhs_info.n0 != 3), "Only 2,3,4,8,16 are supported for n0");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((gemm_info.reinterpret_input_as_3d || gemm_info.depth_output_gemm3d != 0) && (src2 != nullptr)
                                    && (!gemm_info.broadcast_bias),
                                    "Bias addition only supported with broadcast mode in case the input or dst has to be reinterpreted as 3D");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.fp_mixed_precision && (src0->data_type() == DataType::F32), "Mixed precision only supported for F16 data type");
    ARM_COMPUTE_RETURN_ON_ERROR(gemm::validate_image2d_support_on_rhs(*src1, rhs_info));
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!post_op_utils.is_post_op_sequence_supported(gemm_info.post_ops), "The sequence of Post Ops is not supported");

    const unsigned int m = gemm_info.m;
    const unsigned int n = gemm_info.n;
    const unsigned int k = gemm_info.k;

    TensorShape tensor_shape0{ src0->tensor_shape() };
    tensor_shape0.set(0, k);
    tensor_shape0.set(1, m);

    TensorShape tensor_shape1{ src1->tensor_shape() };
    tensor_shape1.set(0, n);
    tensor_shape1.set(1, k);

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

    const TensorInfo tensor_info0 = src0->clone()->set_tensor_shape(tensor_shape0);
    const TensorInfo tensor_info1 = src1->clone()->set_tensor_shape(tensor_shape1);

    const TensorInfo tensor_info_reshaped0 = src0->clone()->set_tensor_shape(misc::shape_calculator::compute_lhs_reshaped_shape(tensor_info0, lhs_info));
    const TensorInfo tensor_info_reshaped1 = src1->clone()->set_tensor_shape(misc::shape_calculator::compute_rhs_reshaped_shape(tensor_info1, rhs_info));

    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src0, &tensor_info_reshaped0);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src1, &tensor_info_reshaped1);

    if(dst->total_size() != 0)
    {
        const TensorInfo tensor_info_dst = dst->clone()->set_tensor_shape(misc::shape_calculator::compute_mm_shape(*src0, *src1, gemm_info));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(dst, &tensor_info_dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src0, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(!post_op_utils.are_post_op_shapes_compliant(dst, gemm_info.post_ops), "The Post Op shapes are not compliant");
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *src0, ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst, const GEMMLHSMatrixInfo &lhs_info,
                                                        const GEMMRHSMatrixInfo &rhs_info,
                                                        const GEMMKernelInfo &gemm_info, ElementsProcessed &num_elements_processed)
{
    ARM_COMPUTE_UNUSED(src0, src1, src2);
    unsigned int &num_elems_processed_per_iteration_x = num_elements_processed[0];
    unsigned int &num_elems_processed_per_iteration_y = num_elements_processed[1];
    bool          reinterpret_output_as_3d            = gemm_info.depth_output_gemm3d != 0;

    TensorInfo tmp_info(*dst);

    if(reinterpret_output_as_3d)
    {
        // Since the dst tensor has to be reinterpreted as 3D and the execute window is based on a 2D GEMM,
        // the window needs to be constructed on the 2D collapsed version of the tensor
        TensorShape tmp_shape(dst->tensor_shape());
        tmp_shape.collapse(2U, 1U);
        tmp_info.set_tensor_shape(tmp_shape);
    }

    // Configure kernel window
    num_elems_processed_per_iteration_x = rhs_info.n0;
    num_elems_processed_per_iteration_y = lhs_info.m0;

    Window win = calculate_max_window(tmp_info, Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));

    // Collapse along the Z direction
    // This collapse needs to be here in order to tune the Z dimension of LWS
    Window             collapsed             = win;
    const unsigned int dimension_to_collapse = std::min(static_cast<unsigned int>(dst->num_dimensions()), 2u);
    collapsed                                = win.collapse(win, dimension_to_collapse);

    return std::make_pair(Status{}, collapsed);
}
} // namespace

ClGemmMatrixMultiplyReshapedKernel::ClGemmMatrixMultiplyReshapedKernel()
{
    _type = CLKernelType::GEMM;
}

void ClGemmMatrixMultiplyReshapedKernel::configure(const CLCompileContext &compile_context,
                                                   const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *src2, ITensorInfo *dst, float alpha, float beta,
                                                   const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info, const GEMMKernelInfo &gemm_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src0, src1, dst);

    // dst tensor auto initialization if not yet initialized
    auto_init_if_empty(*dst, src0->clone()->set_tensor_shape(misc::shape_calculator::compute_mm_shape(*src0, *src1, gemm_info)));

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src0, src1, src2, dst, alpha, beta, lhs_info, rhs_info, gemm_info));

    auto padding_info         = get_padding_info({ src0, src1, src2, dst });
    _reinterpret_output_as_3d = gemm_info.depth_output_gemm3d != 0;
    _use_dummy_work_items     = preferred_dummy_work_items_support(CLKernelLibrary::get().get_device());
    _add_bias                 = src2 != nullptr;
    _export_to_cl_image       = rhs_info.export_to_cl_image;
    _num_post_op_args         = gemm_info.post_ops.total_num_arguments();

    // Check if we need to slide the matrix B
    const unsigned int num_dimensions_src0 = src0->num_dimensions();
    _slide_matrix_b                        = (src1->num_dimensions() >= num_dimensions_src0);

    ElementsProcessed num_elements_processed{};

    // Configure kernel window
    auto win_config = validate_and_configure_window(src0->clone().get(),
                                                    src1->clone().get(),
                                                    (src2 != nullptr) ? src2->clone().get() : nullptr,
                                                    dst->clone().get(),
                                                    lhs_info,
                                                    rhs_info,
                                                    gemm_info,
                                                    num_elements_processed);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

    const bool     enable_mixed_precision = gemm_info.fp_mixed_precision;
    const DataType data_type              = src0->data_type();

    // Calculate partial (store instead of load) M0 and partial N0 for the partial blocks at the end of a row/column if any. This is to avoid padding.
    const unsigned int internal_m = _reinterpret_output_as_3d ? gemm_info.m : dst->dimension(1);

    const unsigned int partial_store_m0 = internal_m % lhs_info.m0;
    const unsigned int partial_store_n0 = gemm_info.n % rhs_info.n0;
    _m                                  = gemm_info.m;
    _n                                  = gemm_info.n;
    _k                                  = gemm_info.k;

    // Create build options
    CLBuildOptions build_opts;
    build_opts.add_option_if(!(helpers::float_ops::is_one(alpha)), "-DALPHA=" + float_to_string_with_full_precision(alpha));
    build_opts.add_option_if(src2 != nullptr, "-DBETA=" + float_to_string_with_full_precision(beta));
    build_opts.add_option_if(helpers::float_ops::is_one(beta), "-DUNIT_BETA");
    build_opts.add_option_if(_reinterpret_output_as_3d, "-DREINTERPRET_OUTPUT_AS_3D");
    build_opts.add_option_if(_reinterpret_output_as_3d, "-DHEIGHT_GEMM3D=" + support::cpp11::to_string(dst->dimension(1)));
    build_opts.add_option_if(_reinterpret_output_as_3d, "-DDEPTH_GEMM3D=" + support::cpp11::to_string(dst->dimension(2)));
    build_opts.add_option_if(gemm_info.broadcast_bias, "-DBROADCAST_BIAS");
    build_opts.add_option_if(!_slide_matrix_b, "-DMATRIX_B_DEPTH=" + support::cpp11::to_string(src1->dimension(2)));
    build_opts.add_option_if(lhs_info.interleave, "-DLHS_INTERLEAVE");
    build_opts.add_option_if(rhs_info.interleave, "-DRHS_INTERLEAVE");
    build_opts.add_option_if(lhs_info.transpose, "-DLHS_TRANSPOSE");
    build_opts.add_option_if(_use_dummy_work_items, "-DDUMMY_WORK_ITEMS");
    build_opts.add_option_if(enable_mixed_precision, "-DMIXED_PRECISION");
    build_opts.add_option_if(rhs_info.export_to_cl_image, "-DOPENCL_IMAGE_SUPPORT");
    build_opts.add_option("-DRHS_HEIGHT=" + support::cpp11::to_string(src1->dimension(1)));
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type));
    build_opts.add_option("-DDATA_TYPE_ACCUMULATOR=" + (enable_mixed_precision ? get_cl_type_from_data_type(DataType::F32) : get_cl_type_from_data_type(data_type)));
    build_opts.add_option("-DM0=" + support::cpp11::to_string(lhs_info.m0));
    build_opts.add_option("-DN0=" + support::cpp11::to_string(rhs_info.n0));
    build_opts.add_option("-DK0=" + support::cpp11::to_string(lhs_info.k0));
    build_opts.add_option("-DV0=" + support::cpp11::to_string(lhs_info.v0));
    build_opts.add_option("-DH0=" + support::cpp11::to_string(rhs_info.h0));
    build_opts.add_option("-DPARTIAL_STORE_M0=" + support::cpp11::to_string(partial_store_m0));
    build_opts.add_option("-DPARTIAL_STORE_N0=" + support::cpp11::to_string(partial_store_n0));
    // If post_ops are used, then we disable the use of gemm_info.activation_info
    if(gemm_info.post_ops.size() > 0)
    {
        post_op_utils.set_post_ops_cl_build_options(build_opts, gemm_info.post_ops);
    }
    else
    {
        build_opts.add_option_if(gemm_info.activation_info.enabled(), "-DACTIVATION_TYPE=" + lower_string(string_from_activation_func(gemm_info.activation_info.activation())));
        build_opts.add_option_if(gemm_info.activation_info.enabled(), "-DA_VAL=" + float_to_string_with_full_precision(gemm_info.activation_info.a()));
        build_opts.add_option_if(gemm_info.activation_info.enabled(), "-DB_VAL=" + float_to_string_with_full_precision(gemm_info.activation_info.b()));
    }

    std::string kernel_name("gemm_mm_reshaped_");
    kernel_name += lhs_info.transpose ? "lhs_t_" : "lhs_nt_";
    kernel_name += rhs_info.transpose ? "rhs_t" : "rhs_nt";
    kernel_name += rhs_info.export_to_cl_image ? "_texture" : "";
    post_op_utils.set_post_ops_cl_kernel_name(kernel_name, gemm_info.post_ops);

    // A macro guard to compile ONLY the kernel of interest
    build_opts.add_option("-D" + upper_string(kernel_name));

    // Create kernel
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += (_add_bias ? "add_bias_" : "");
    _config_id += (gemm_info.broadcast_bias ? "broadcast_bias_" : "");
    _config_id += (_reinterpret_output_as_3d ? "3do_" : "");
    _config_id += (gemm_info.activation_info.enabled() ? "fused_activation_" : "");
    _config_id += lower_string(string_from_data_type(src0->data_type()));
    _config_id += "_";
    _config_id += (enable_mixed_precision ? "mixed_precision_" : "");
    _config_id += support::cpp11::to_string(dst->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(gemm_info.k);
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(2));
    _config_id += "_";
    _config_id += support::cpp11::to_string(lhs_info.m0);
    _config_id += "_";
    _config_id += support::cpp11::to_string(rhs_info.n0);
    _config_id += "_";
    _config_id += support::cpp11::to_string(lhs_info.k0);
    _config_id += "_";
    _config_id += support::cpp11::to_string(lhs_info.v0);
    _config_id += "_";
    _config_id += support::cpp11::to_string(rhs_info.h0);
    _config_id += "_";
    _config_id += support::cpp11::to_string(lhs_info.interleave);
    _config_id += "_";
    _config_id += support::cpp11::to_string(rhs_info.interleave);

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status ClGemmMatrixMultiplyReshapedKernel::validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, float alpha, float beta,
                                                    const GEMMLHSMatrixInfo &lhs_info,
                                                    const GEMMRHSMatrixInfo &rhs_info, const GEMMKernelInfo &gemm_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src0, src1, src2, dst, alpha, beta, lhs_info, rhs_info, gemm_info));
    return Status{};
}

void ClGemmMatrixMultiplyReshapedKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
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

    Window slice          = window.first_slice_window_3D();
    Window slice_matrix_b = slice;

    slice_matrix_b.set(Window::DimX, Window::Dimension(0, 1, 1));
    slice_matrix_b.set(Window::DimY, Window::Dimension(0, 1, 1));

    const unsigned int total_cross_plane_pad = dst->info()->padding().top + dst->info()->padding().bottom;

    cl::Image2D src1_image2d;

    if(_export_to_cl_image)
    {
        const TensorShape shape2d(src1->info()->dimension(0) / 4, src1->info()->dimension(1) * src1->info()->dimension(2));
        const size_t      image_row_pitch = src1->info()->strides_in_bytes()[1];

        src1_image2d = create_image2d_from_buffer(CLKernelLibrary::get().context(), src1->cl_buffer(), shape2d, src1->info()->data_type(), image_row_pitch, CLImage2DType::ReadOnly);
    }

    do
    {
        Window slice_b = slice;
        // Don't slice matrix B along the z dimension if matrix B has just 2 dimensions and matrix A more than 2
        // This scenario can happen when the matrix multiplication is used to perform a convolution operation
        if(!_slide_matrix_b)
        {
            slice_b = slice_matrix_b;
        }

        unsigned int idx = 0;

        // LHS buffer
        add_2D_tensor_argument(idx, src0, slice);

        // RHS buffer or RHS OpenCL image (_export_to_cl_image == true)
        if(_export_to_cl_image)
        {
            _kernel.setArg(idx++, src1_image2d);
        }
        else
        {
            add_2D_tensor_argument(idx, src1, slice_b);
        }

        // Bias buffer (_add_bias == true)
        add_2D_tensor_argument_if(_add_bias, idx, src2, slice);

        // dst buffer
        add_2D_tensor_argument(idx, dst, slice);

        // post op argument buffers
        for(size_t i = 0; i < _num_post_op_args; ++i)
        {
            const auto post_op_arg = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(experimental::get_post_op_arg_type(i)));
            add_2D_tensor_argument(idx, post_op_arg, slice);
        }

        // LHS stride_z
        _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(src0->info()->strides_in_bytes()[2]));

        // RHS stride_z (not used if _export_to_cl_image == true)
        _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(src1->info()->strides_in_bytes()[2]));

        // Bias stride_z (if _add_bias == true)
        if(_add_bias)
        {
            _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(src2->info()->strides_in_bytes()[2]));
        }

        // dst stride_z
        _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(dst->info()->strides_in_bytes()[2]));

        // post op argument stride_z
        for(size_t i = 0; i < _num_post_op_args; ++i)
        {
            const auto post_op_arg = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(experimental::get_post_op_arg_type(i)));
            _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(post_op_arg->info()->strides_in_bytes()[2]));
        }
        // Cross-plan padding (if _reinterpret_output_as_3d = true)
        if(_reinterpret_output_as_3d)
        {
            _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(total_cross_plane_pad));
        }

        // Pass m, n and k at runtime
        _kernel.setArg<cl_int>(idx++, _m);
        _kernel.setArg<cl_int>(idx++, _n);

        // K dimension (not used if _export_to_cl_image == true)
        _kernel.setArg<cl_int>(idx++, _k);

        // Dispatch kernel
        enqueue(queue, *this, slice, lws_hint(), _use_dummy_work_items);
    }
    while(window.slide_window_slice_3D(slice));
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
