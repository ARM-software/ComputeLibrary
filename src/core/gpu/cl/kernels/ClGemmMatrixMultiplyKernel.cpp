/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "src/core/gpu/cl/kernels/ClGemmMatrixMultiplyKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/core/utils/helpers/float_ops.h"
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

inline Status validate_arguments(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, float beta,
                                 bool is_interleaved_transposed, const GEMMReshapeInfo &reshape_info, bool fp_mixed_precision)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src0, src1, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src0);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src0, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src0, src1);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((fp_mixed_precision && (src0->data_type() != DataType::F16)), "Mixed precision floating point is supported only for F16 data");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(src0->num_dimensions() > 4, "The number of dimensions for the matrix A must be <= 4");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(src1->num_dimensions() > 3, "The number of dimensions for the matrix B must be <= 3");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(is_interleaved_transposed && reshape_info.reinterpret_input_as_3d(), "The input tensor cannot be reinterpreted as 3D if is_interleaved_transposed is true");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(src1->num_dimensions() > 2 && reshape_info.reinterpret_input_as_3d(), "The src1 tensor cannot have more than 2 dimensions if src0 has to be reinterpreted as 3D");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((reshape_info.reinterpret_input_as_3d() || reshape_info.depth_output_gemm3d() != 0) && (src2 != nullptr)
                                    && (!reshape_info.broadcast_bias()),
                                    "Bias addition only supported with broadcast mode in case the input or dst has to be reinterpreted as 3D");

    if(!is_interleaved_transposed)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(src0->dimension(0) != src1->dimension(1));

        if(src2 != nullptr && !(helpers::float_ops::is_zero(beta)))
        {
            const unsigned int m         = reshape_info.reinterpret_input_as_3d() ? src0->dimension(1) * src0->dimension(2) : src0->dimension(1);
            const unsigned int n         = src1->dimension(0);
            const unsigned int src2_dim0 = src2->dimension(0);
            const unsigned int src2_dim1 = src2->dimension(1);

            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src2, src1);
            if(reshape_info.broadcast_bias())
            {
                ARM_COMPUTE_RETURN_ERROR_ON_MSG((src2_dim1 != 1 || src2_dim0 != n), "Incorrect dimension of bias matrix which is to be broadcasted");
            }
            else
            {
                ARM_COMPUTE_RETURN_ERROR_ON_MSG((src2_dim0 != n || src2_dim1 != m), "Incorrect dimension of bias matrix");
            }
        }
    }
    else
    {
        GEMMRHSMatrixInfo rhs_info;
        GEMMLHSMatrixInfo lhs_info;
        const auto        m                         = static_cast<unsigned int>(reshape_info.m());
        const auto        n                         = static_cast<unsigned int>(reshape_info.n());
        const int         k                         = reshape_info.k();
        const int         mult_transpose1xW_width   = reshape_info.mult_transpose1xW_width();
        const int         mult_interleave4x4_height = reshape_info.mult_interleave4x4_height();
        rhs_info.n0                                 = max_cl_vector_width / src1->element_size();
        rhs_info.k0                                 = 1;
        rhs_info.h0                                 = mult_transpose1xW_width;
        rhs_info.interleave                         = false;
        rhs_info.transpose                          = false;
        lhs_info.m0                                 = 4;
        lhs_info.k0                                 = 4;
        lhs_info.v0                                 = mult_interleave4x4_height;
        lhs_info.interleave                         = true;
        lhs_info.transpose                          = true;

        TensorShape tensor_shape0{ src0->tensor_shape() };
        tensor_shape0.set(0, k);
        tensor_shape0.set(1, m);

        TensorShape tensor_shape1{ src1->tensor_shape() };
        tensor_shape1.set(0, n);
        tensor_shape1.set(1, k);

        const TensorInfo tensor_info0 = src0->clone()->set_tensor_shape(tensor_shape0);
        const TensorInfo tensor_info1 = src1->clone()->set_tensor_shape(tensor_shape1);

        const TensorInfo tensor_info_reshaped0 = src0->clone()->set_tensor_shape(misc::shape_calculator::compute_lhs_reshaped_shape(tensor_info0, lhs_info));
        const TensorInfo tensor_info_reshaped1 = src1->clone()->set_tensor_shape(misc::shape_calculator::compute_rhs_reshaped_shape(tensor_info1, rhs_info));

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src0, &tensor_info_reshaped0);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src1, &tensor_info_reshaped1);

        if(src2 != nullptr && !(helpers::float_ops::is_zero(beta)))
        {
            const unsigned int src2_dim0 = src2->dimension(0);
            const unsigned int src2_dim1 = src2->dimension(1);

            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src2, src1);
            if(reshape_info.broadcast_bias())
            {
                ARM_COMPUTE_RETURN_ERROR_ON_MSG((src2_dim1 != 1 || src2_dim0 != n), "Incorrect dimension of bias matrix which is to be broadcasted");
            }
            else
            {
                ARM_COMPUTE_RETURN_ERROR_ON_MSG((src2_dim0 != n || src2_dim1 != m), "Incorrect dimension of bias matrix");
            }
        }
    }

    if(dst->total_size() != 0)
    {
        const TensorInfo tensor_info_dst = dst->clone()->set_tensor_shape(misc::shape_calculator::compute_mm_shape(*src0, *src1, is_interleaved_transposed, reshape_info));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(dst, &tensor_info_dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src0, dst);
    }

    return Status{};
}

inline std::pair<Status, Window> validate_and_configure_window(ITensorInfo *src0, ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst,
                                                               float beta, bool is_interleaved_transposed, const GEMMReshapeInfo &reshape_info, GPUTarget gpu_target,
                                                               ElementsProcessed &num_elements_processed)
{
    ARM_COMPUTE_UNUSED(beta);
    bool   window_changed = false;
    Window win{};
    Window win_out{};

    const DataType data_type                           = src0->data_type();
    unsigned int &num_elems_processed_per_iteration_x = num_elements_processed[0];
    unsigned int &num_elems_processed_per_iteration_y = num_elements_processed[1];
    bool           reinterpret_input_as_3d             = reshape_info.reinterpret_input_as_3d();
    bool           reinterpret_output_as_3d            = (reshape_info.depth_output_gemm3d() != 0);

    // In case both input and dst have to be reinterpreted as 3D tensors,
    // force reinterpret_input_as_3d and reinterpret_output_as_3d to be false.
    if(reinterpret_input_as_3d == reinterpret_output_as_3d)
    {
        reinterpret_input_as_3d  = false;
        reinterpret_output_as_3d = false;
    }

    // dst tensor auto inizialitation if not yet initialized
    auto_init_if_empty(*dst, src0->clone()->set_tensor_shape(misc::shape_calculator::compute_mm_shape(*src0, *src1, is_interleaved_transposed, reshape_info)));

    TensorInfo tmp_info(*dst);

    if(reinterpret_output_as_3d)
    {
        // Since the dst tensor has to be reinterpreted as 3D and the execute window is based on a 2D GEMM,
        // the window needs to be constructed on the 2D collapsed version of the tensor
        TensorShape tmp_shape(dst->tensor_shape());
        tmp_shape.collapse(2U, 1U);
        tmp_info.set_tensor_shape(tmp_shape);
    }

    if(is_interleaved_transposed)
    {
        // reinterpret_input_as_3d is not supported if is_interleaved_transposed is set
        ARM_COMPUTE_ERROR_ON(reshape_info.reinterpret_input_as_3d());

        // Configure kernel window
        num_elems_processed_per_iteration_x = max_cl_vector_width / data_size_from_type(data_type);
        num_elems_processed_per_iteration_y = 4;

        win = calculate_max_window(tmp_info, Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));
        if(src2 != nullptr)
        {
            const int bias_processed_per_iteration_x = num_elems_processed_per_iteration_x;

            const int bias_processed_per_iteration_y = reshape_info.broadcast_bias() ? 1 : num_elems_processed_per_iteration_y;

            AccessWindowStatic src2_access(src2, 0, 0,
                                           ceil_to_multiple(src2->dimension(0), bias_processed_per_iteration_x),
                                           ceil_to_multiple(src2->dimension(1), bias_processed_per_iteration_y));

            window_changed = update_window_and_padding(win, src2_access); // window used by the execute_window_loop
        }
    }
    else // The input tensors have not been reshaped
    {
        // Special case for 1xN, 2xN, 3xN and 4xN src0 tensor. num_elems_processed_per_iteration_x is set up for the default case.
        num_elems_processed_per_iteration_x = max_cl_vector_width / data_size_from_type(data_type);
        num_elems_processed_per_iteration_y = std::min(static_cast<int>(dst->dimension(1)), 4);

        // Create kernels according to the architecture, data type and input size.
        GPUTarget arch_target = get_arch_from_target(gpu_target);
        if(arch_target == GPUTarget::BIFROST && data_type == DataType::F32)
        {
            num_elems_processed_per_iteration_x = (src1->dimension(0) <= 1000 && src0->num_dimensions() == 1) ? 2 : 4;
        }

        // Configure window
        win     = calculate_max_window(tmp_info, Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));
        win_out = calculate_max_window(*dst, Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));
        AccessWindowStatic src0_access(src0, 0, 0, src0->dimension(0), src0->dimension(1));
        AccessWindowStatic src1_access(src1, 0, 0, ceil_to_multiple(src1->dimension(0), num_elems_processed_per_iteration_x), src1->dimension(1));
        AccessWindowStatic dst_access(dst, 0, 0,
                                      dst->dimension(0),
                                      dst->dimension(1));

        if(src2 != nullptr)
        {
            const int bias_processed_per_iteration_x = num_elems_processed_per_iteration_x;

            AccessWindowStatic src2_access(src2, 0, 0,
                                           ceil_to_multiple(src2->dimension(0), bias_processed_per_iteration_x),
                                           src2->dimension(1));

            window_changed = update_window_and_padding(win, src0_access, src1_access, src2_access) || // window used by the execute_window_loop
                             update_window_and_padding(win_out, dst_access);                          // window used to update the padding requirements of dst tensor
        }
        else
        {
            window_changed = update_window_and_padding(win, src0_access, src1_access) || // window used by the execute_window_loop
                             update_window_and_padding(win_out, dst_access);             // window used to update the padding requirements of dst tensor
        }
    }

    // Collapse along the Z direction
    // This collapse needs to be here in order to tune the Z dimension of LWS
    Window             collapsed             = win;
    const unsigned int dimension_to_collapse = std::min(static_cast<unsigned int>(dst->num_dimensions()), 2u);
    collapsed                                = win.collapse(win, dimension_to_collapse);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, collapsed);
}
} // namespace

ClGemmMatrixMultiplyKernel::ClGemmMatrixMultiplyKernel()
{
    _type = CLKernelType::GEMM;
}

void ClGemmMatrixMultiplyKernel::configure(const CLCompileContext &compile_context, ITensorInfo *src0, ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst, float alpha,
                                           float beta,
                                           bool is_interleaved_transposed, const GEMMReshapeInfo &reshape_info, bool fp_mixed_precision, const ActivationLayerInfo &activation_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src0, src1, dst);

    // Perform validate step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src0, src1, src2, dst, beta,
                                                  is_interleaved_transposed, reshape_info, fp_mixed_precision));

    auto padding_info = is_interleaved_transposed ? get_padding_info({ src0, src1, dst }) : get_padding_info({ src0, dst });

    _reinterpret_input_as_3d  = reshape_info.reinterpret_input_as_3d();
    _reinterpret_output_as_3d = (reshape_info.depth_output_gemm3d() != 0);
    _add_bias                 = src2 != nullptr;

    // In case both input and dst have to be reinterpreted as 3D tensors,
    // force reinterpret_input_as_3d and reinterpret_output_as_3d to be false.
    if(_reinterpret_input_as_3d == _reinterpret_output_as_3d)
    {
        _reinterpret_input_as_3d  = false;
        _reinterpret_output_as_3d = false;
    }

    // Check if we need to slide the matrix B
    const unsigned int num_dimensions_src0 = _reinterpret_input_as_3d ? src0->num_dimensions() - 1 : src0->num_dimensions();

    _slide_matrix_b = (src1->num_dimensions() >= num_dimensions_src0);

    const DataType data_type = src0->data_type();

    // Get target architecture
    GPUTarget gpu_target = get_target();

    ElementsProcessed num_elements_processed{};

    // Configure kernel window
    auto win_config = validate_and_configure_window(src0, src1, src2, dst, beta, is_interleaved_transposed, reshape_info,
                                                    gpu_target, num_elements_processed);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

    // If _reinterpret_input_as_3d = _reinterpret_output_as_3d = true, both will be turned off (false)
    // in which case we will dispatch a batched-GEMM to reduce the complexity of the address calculation within the OpenCL kernel.
    // This means that the actual m used by the kernel is given by dst->dimension(1)
    const unsigned int internal_m = _reinterpret_output_as_3d ? dst->dimension(1) * dst->dimension(2) : dst->dimension(1);
    const unsigned int n          = dst->dimension(0);

    const unsigned int h_gemm_3d = _reinterpret_output_as_3d ? dst->dimension(1) : src0->dimension(1);
    const unsigned int d_gemm_3d = _reinterpret_output_as_3d ? dst->dimension(2) : src0->dimension(2);

    const unsigned int m0 = num_elements_processed.y();
    const unsigned int n0 = num_elements_processed.x();

    // Calculate partial (store instead of load) M0 and partial N0 for the partial blocks at the end of a row/column if any. This is to avoid padding.
    const unsigned int partial_store_m0 = internal_m % m0;
    const unsigned int partial_store_n0 = n % n0;

    // Create build options
    CLBuildOptions build_opts;

    build_opts.add_option_if(!(helpers::float_ops::is_one(alpha)), "-DALPHA=" + float_to_string_with_full_precision(alpha));
    build_opts.add_option_if(src2 != nullptr, "-DBETA=" + float_to_string_with_full_precision(beta));
    build_opts.add_option_if(helpers::float_ops::is_one(beta), "-DUNIT_BETA");
    build_opts.add_option_if(reshape_info.broadcast_bias(), "-DBROADCAST_BIAS");
    build_opts.add_option_if(_reinterpret_input_as_3d, "-DREINTERPRET_INPUT_AS_3D");
    build_opts.add_option_if(_reinterpret_output_as_3d, "-DREINTERPRET_OUTPUT_AS_3D");
    build_opts.add_option_if(_reinterpret_input_as_3d || _reinterpret_output_as_3d, "-DHEIGHT_GEMM3D=" + support::cpp11::to_string(h_gemm_3d));
    build_opts.add_option_if(_reinterpret_input_as_3d || _reinterpret_output_as_3d, "-DDEPTH_GEMM3D=" + support::cpp11::to_string(d_gemm_3d));
    build_opts.add_option_if(!_slide_matrix_b, "-DMATRIX_B_DEPTH=" + support::cpp11::to_string(src1->dimension(2)));
    build_opts.add_option_if(activation_info.enabled(), "-DACTIVATION_TYPE=" + lower_string(string_from_activation_func(activation_info.activation())));
    build_opts.add_option_if(activation_info.enabled(), "-DA_VAL=" + float_to_string_with_full_precision(activation_info.a()));
    build_opts.add_option_if(activation_info.enabled(), "-DB_VAL=" + float_to_string_with_full_precision(activation_info.b()));
    build_opts.add_option("-DIN1_DIM_X=" + support::cpp11::to_string(src1->dimension(0)));

    const bool is_bifrost = get_arch_from_target(gpu_target) == GPUTarget::BIFROST;

    std::string kernel_name;
    if(is_interleaved_transposed)
    {
        const int mult_transpose1xW_width   = reshape_info.mult_transpose1xW_width();
        const int mult_interleave4x4_height = reshape_info.mult_interleave4x4_height();

        build_opts.add_option("-DM=" + support::cpp11::to_string(internal_m));
        build_opts.add_option("-DN=" + support::cpp11::to_string(n));
        build_opts.add_option("-DK=" + support::cpp11::to_string(src1->dimension(0) / (n0 * mult_transpose1xW_width)));
        build_opts.add_option("-DH0=" + support::cpp11::to_string(mult_transpose1xW_width));
        build_opts.add_option("-DV0=" + support::cpp11::to_string(mult_interleave4x4_height));
        build_opts.add_option("-DPARTIAL_STORE_M0=" + support::cpp11::to_string(partial_store_m0));
        build_opts.add_option("-DPARTIAL_STORE_N0=" + support::cpp11::to_string(partial_store_n0));

        if(is_data_type_float(data_type) && is_bifrost)
        {
            kernel_name = "gemm_mm_interleaved_transposed_" + lower_string(string_from_data_type(data_type)) + "_bifrost";
        }
        else
        {
            kernel_name = "gemm_mm_interleaved_transposed_" + lower_string(string_from_data_type(data_type));
            if(fp_mixed_precision && data_type == DataType::F16)
            {
                // currently wider accumulator is only supported for fp16 kernels.
                kernel_name += "_acc32";
            }
        }
    }
    else // The input tensors have not been reshaped
    {
        build_opts.add_option("-DN=" + support::cpp11::to_string(n));
        build_opts.add_option("-DK=" + support::cpp11::to_string(src0->dimension(0)));
        build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type));
        build_opts.add_option("-DM0=" + support::cpp11::to_string(m0));
        build_opts.add_option("-DN0=" + support::cpp11::to_string(n0));
        build_opts.add_option("-DPARTIAL_STORE_M0=" + support::cpp11::to_string(partial_store_m0));
        build_opts.add_option("-DPARTIAL_STORE_N0=" + support::cpp11::to_string(partial_store_n0));

        // Create kernels according to the architecture, data type and input size.
        if(is_data_type_float(data_type) && is_bifrost)
        {
            kernel_name = "gemm_mm_floating_point";

            if(src0->num_dimensions() != 1)
            {
                kernel_name += "_" + lower_string(string_from_data_type(data_type)) + "_bifrost";
                if(fp_mixed_precision && data_type == DataType::F16)
                {
                    // currently wider accumulator is only supported for fp16 kernels.
                    kernel_name += "_acc32";
                }
            }
            else if(src1->dimension(0) <= 1000 && data_type == DataType::F32)
            {
                // The first kernel is optimized for the case of 1000 or less dst elements (e.g. FC8 of AlexNet and VGG-16, and
                // FC1 of Inception v3). The second kernel is optimized for the case of greater than 1000 dst elements (e.g.
                // FC6 and FC7 of AlexNet and VGG-16).
                kernel_name += "_" + lower_string(string_from_data_type(data_type)) + "_bifrost_1000";
            }

            // The work-group size equal to the Bifrost quad size has been proved to be optimal for these kernels
            // via exhaustive autotuning over a range of representative layer configurations.
            set_lws_hint(cl::NDRange(4));
        }
        else // (MIDGARD and F32) or (F16)
        {
            kernel_name = "gemm_mm_floating_point";
        }
    }
    // Create kernel
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Set config_id for enabling LWS tuning
    _config_id = "gemm_";
    _config_id += (is_interleaved_transposed ? "reshaped_" : "");
    _config_id += (_add_bias ? "add_bias_" : "");
    _config_id += (reshape_info.broadcast_bias() ? "broadcast_bias_" : "");
    _config_id += (fp_mixed_precision ? "fp_mixed_" : "");
    _config_id += (_reinterpret_input_as_3d ? "3di_" : "");
    _config_id += (_reinterpret_output_as_3d ? "3do_" : "");
    _config_id += lower_string(string_from_data_type(src0->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(2));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(3));
    _config_id += "_";
    _config_id += (is_interleaved_transposed ? support::cpp11::to_string(src1->dimension(0)) : support::cpp11::to_string(src1->dimension(1)));

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status ClGemmMatrixMultiplyKernel::validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, float alpha, float beta,
                                            bool is_interleaved_transposed, const GEMMReshapeInfo &reshape_info, GPUTarget gpu_target, bool fp_mixed_precision, const ActivationLayerInfo &activation_info)
{
    // Note: num_elements_processed will be set in validate_and_configure_window()
    ElementsProcessed num_elements_processed{};
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_UNUSED(activation_info);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src0, src1, src2, dst, beta, is_interleaved_transposed, reshape_info, fp_mixed_precision));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(src0->clone().get(),
                                                              src1->clone().get(),
                                                              (src2 != nullptr) ? src2->clone().get() : nullptr,
                                                              dst->clone().get(),
                                                              beta,
                                                              is_interleaved_transposed,
                                                              reshape_info,
                                                              gpu_target,
                                                              num_elements_processed)
                                .first);

    return Status{};
}

void ClGemmMatrixMultiplyKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
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

    const unsigned int num_arguments_bias = _add_bias ? num_arguments_per_2D_tensor() + 1 : 0;

    if(_reinterpret_input_as_3d)
    {
        // Pass bottom paddings to the kernel if the input has to be reinterpreted as 3D tensor
        const unsigned int idx0                  = 3 * num_arguments_per_2D_tensor() + 3 + num_arguments_bias;
        const unsigned int total_cross_plane_pad = src0->info()->padding().top + src0->info()->padding().bottom;
        _kernel.setArg<cl_uint>(idx0, static_cast<unsigned int>(total_cross_plane_pad));
    }

    if(_reinterpret_output_as_3d)
    {
        // Pass bottom paddings to the kernel if the dst has to be reinterpreted as 3D tensor
        const unsigned int idx0                  = 3 * num_arguments_per_2D_tensor() + 3 + (_reinterpret_input_as_3d ? 1 : 0) + num_arguments_bias;
        const unsigned int total_cross_plane_pad = dst->info()->padding().top + dst->info()->padding().bottom;
        _kernel.setArg<cl_uint>(idx0, static_cast<unsigned int>(total_cross_plane_pad));
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
        add_2D_tensor_argument(idx, src0, slice);
        add_2D_tensor_argument(idx, src1, slice_b);
        if(_add_bias)
        {
            add_2D_tensor_argument(idx, src2, slice);
        }
        add_2D_tensor_argument(idx, dst, slice);
        _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(src0->info()->strides_in_bytes()[2]));
        _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(src1->info()->strides_in_bytes()[2]));
        if(_add_bias)
        {
            _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(src2->info()->strides_in_bytes()[2]));
        }
        _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(dst->info()->strides_in_bytes()[2]));
        enqueue(queue, *this, slice, lws_hint());
    }
    while(window.slide_window_slice_3D(slice));
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
