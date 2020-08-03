/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#include "arm_compute/core/CL/kernels/CLGEMMMatrixMultiplyReshapedOnlyRHSKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/gemm/CLGEMMHelpers.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/helpers/float_ops.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/CL/CLUtils.h"
#include "support/StringSupport.h"

#include <tuple>

using namespace arm_compute::misc::shape_calculator;

namespace arm_compute
{
namespace
{
using ElementsProcessed = Steps;

Status validate_arguments(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, float alpha, float beta, const GEMMLHSMatrixInfo &lhs_info,
                          const GEMMRHSMatrixInfo &rhs_info,
                          const GEMMKernelInfo    &gemm_info)
{
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input0, input1, output);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input0);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input0, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input0, input1);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input0->num_dimensions() > 4, "The number of dimensions for the LHS matrix must be <= 4");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input1->num_dimensions() > 3, "The number of dimensions for the RHS matrix must be <= 3");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(lhs_info.m0 < 1 || lhs_info.m0 > 8, "Only 1,2,3,4,5,6,7,8 are supported for m0");
    ARM_COMPUTE_RETURN_ERROR_ON(rhs_info.k0 > 16 || rhs_info.k0 < 2);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(((rhs_info.k0 & (rhs_info.k0 - 1)) && rhs_info.k0 != 3), "Only 2,3,4,8,16 are supported for k0");
    ARM_COMPUTE_RETURN_ERROR_ON(rhs_info.n0 > 16 || rhs_info.n0 < 2);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(((rhs_info.n0 & (rhs_info.n0 - 1)) && rhs_info.n0 != 3), "Only 2,3,4,8,16 are supported for n0");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((gemm_info.reinterpret_input_as_3d || gemm_info.depth_output_gemm3d != 0) && (input2 != nullptr)
                                    && (!gemm_info.broadcast_bias),
                                    "Bias addition only supported with broadcast mode in case the input or output has to be reinterpreted as 3D");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.fp_mixed_precision, "Mixed precision not supported");
    ARM_COMPUTE_RETURN_ON_ERROR(cl_gemm::validate_image2d_support_on_rhs(*input1, rhs_info));

    const unsigned int m = gemm_info.m;
    const unsigned int n = gemm_info.n;
    const unsigned int k = gemm_info.k;

    TensorShape tensor_shape1{ input1->tensor_shape() };
    tensor_shape1.set(0, n);
    tensor_shape1.set(1, k);

    if(input2 != nullptr && !(helpers::float_ops::is_zero(beta)))
    {
        const unsigned int input2_dim0 = input2->dimension(0);
        const unsigned int input2_dim1 = input2->dimension(1);

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input2, input0);
        if(gemm_info.broadcast_bias)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MSG((input2_dim1 != 1 || input2_dim0 != n), "Incorrect dimension of bias matrix which is to be broadcasted");
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MSG((input2_dim0 != n || input2_dim1 != m), "Incorrect dimension of bias matrix");
        }
    }

    const TensorInfo tensor_info1 = input1->clone()->set_tensor_shape(tensor_shape1);

    const TensorInfo tensor_info_reshaped1 = input1->clone()->set_tensor_shape(compute_rhs_reshaped_shape(tensor_info1, rhs_info));

    ARM_COMPUTE_RETURN_ERROR_ON(input0->dimension(0) != k);
    if(gemm_info.reinterpret_input_as_3d)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(input0->dimension(1) * input0->dimension(2) != m);
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON(input0->dimension(1) != m);
    }
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input1, &tensor_info_reshaped1);

    if(output->total_size() != 0)
    {
        const TensorInfo tensor_info_output = output->clone()->set_tensor_shape(compute_mm_shape(*input0, *input1, gemm_info));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(output, &tensor_info_output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input0, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input0, ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output, const GEMMLHSMatrixInfo &lhs_info,
                                                        const GEMMRHSMatrixInfo &rhs_info,
                                                        const GEMMKernelInfo &gemm_info, ElementsProcessed &num_elements_processed)
{
    unsigned int &num_elems_processed_per_iteration_x = num_elements_processed[0];
    unsigned int &num_elems_processed_per_iteration_y = num_elements_processed[1];
    bool          reinterpret_input_as_3d             = gemm_info.reinterpret_input_as_3d;
    bool          reinterpret_output_as_3d            = gemm_info.depth_output_gemm3d != 0;

    Window win{};
    Window win_out{};
    bool   window_changed = false;

    // In case both input and output have to be reinterpreted as 3D tensors,
    // force reinterpret_input_as_3d and reinterpret_output_as_3d to be false.
    if(reinterpret_input_as_3d == reinterpret_output_as_3d)
    {
        reinterpret_output_as_3d = false;
    }

    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*output, input0->clone()->set_tensor_shape(compute_mm_shape(*input0, *input1, gemm_info)));

    TensorInfo tmp_info(*output);

    if(reinterpret_output_as_3d)
    {
        // Since the output tensor has to be reinterpreted as 3D and the execute window is based on a 2D GEMM,
        // the window needs to be constructed on the 2D collapsed version of the tensor
        TensorShape tmp_shape(output->tensor_shape());
        tmp_shape.collapse(2U, 1U);
        tmp_info.set_tensor_shape(tmp_shape);
    }

    // Configure kernel window
    num_elems_processed_per_iteration_x = rhs_info.n0;
    num_elems_processed_per_iteration_y = lhs_info.m0;

    win     = calculate_max_window(tmp_info, Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));
    win_out = calculate_max_window(*output, Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));

    AccessWindowStatic input0_access(input0, 0, 0,
                                     input0->dimension(0),
                                     input0->dimension(1));
    AccessWindowStatic input1_access(input1, 0, 0,
                                     ceil_to_multiple(input1->dimension(0), num_elems_processed_per_iteration_x),
                                     input1->dimension(1));
    AccessWindowStatic output_access(output, 0, 0,
                                     output->dimension(0),
                                     output->dimension(1));

    if(input2 != nullptr)
    {
        const int bias_processed_per_iteration_x = num_elems_processed_per_iteration_x;

        AccessWindowStatic input2_access(input2, 0, 0,
                                         ceil_to_multiple(input2->dimension(0), bias_processed_per_iteration_x),
                                         input2->dimension(1));

        window_changed = update_window_and_padding(win, input0_access, input1_access, input2_access) || // window used by the execute_window_loop
                         update_window_and_padding(win_out, output_access);                             // window used to update the padding requirements of output tensor
    }
    else
    {
        window_changed = update_window_and_padding(win, input0_access, input1_access) || // window used by the execute_window_loop
                         update_window_and_padding(win_out, output_access);              // window used to update the padding requirements of output tensor
    }

    output_access.set_valid_region(win_out, ValidRegion(Coordinates(), output->tensor_shape()));

    // Collapse along the Z direction
    // This collapse needs to be here in order to tune the Z dimension of LWS
    Window             collapsed             = win;
    const unsigned int dimension_to_collapse = std::min(static_cast<unsigned int>(output->num_dimensions()), 2u);
    collapsed                                = win.collapse(win, dimension_to_collapse);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, collapsed);
}
} // namespace

CLGEMMMatrixMultiplyReshapedOnlyRHSKernel::CLGEMMMatrixMultiplyReshapedOnlyRHSKernel()
    : _input0(nullptr), _input1(nullptr), _input2(nullptr), _output(nullptr), _slide_matrix_b(true), _reinterpret_input_as_3d(false), _reinterpret_output_as_3d(false), _use_dummy_work_items(false),
      _add_bias(false), _broadcast_bias(false), _export_to_cl_image(false)
{
}

void CLGEMMMatrixMultiplyReshapedOnlyRHSKernel::configure(const ICLTensor *input0, const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, float alpha, float beta,
                                                          const GEMMLHSMatrixInfo &lhs_info,
                                                          const GEMMRHSMatrixInfo &rhs_info, const GEMMKernelInfo &gemm_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input0, input1, input2, output, alpha, beta, lhs_info, rhs_info, gemm_info);
}

void CLGEMMMatrixMultiplyReshapedOnlyRHSKernel::configure(const CLCompileContext &compile_context, const ICLTensor *input0, const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output,
                                                          float                    alpha,
                                                          float                    beta,
                                                          const GEMMLHSMatrixInfo &lhs_info,
                                                          const GEMMRHSMatrixInfo &rhs_info, const GEMMKernelInfo &gemm_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input0, input1, output);

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input0->info(), input1->info(), (input2 != nullptr ? input2->info() : nullptr), output->info(), alpha, beta, lhs_info, rhs_info, gemm_info));

    _input0                   = input0;
    _input1                   = input1;
    _input2                   = helpers::float_ops::is_zero(beta) ? nullptr : input2;
    _output                   = output;
    _reinterpret_input_as_3d  = gemm_info.reinterpret_input_as_3d;
    _reinterpret_output_as_3d = gemm_info.depth_output_gemm3d != 0;
    _use_dummy_work_items     = preferred_dummy_work_items_support(CLKernelLibrary::get().get_device());
    _add_bias                 = _input2 != nullptr;
    _broadcast_bias           = gemm_info.broadcast_bias;
    _export_to_cl_image       = rhs_info.export_to_cl_image;

    // In case both input and output have to be reinterpreted as 3D tensors,
    // force reinterpret_input_as_3d and reinterpret_output_as_3d to be false.
    if(_reinterpret_input_as_3d == _reinterpret_output_as_3d)
    {
        _reinterpret_input_as_3d  = false;
        _reinterpret_output_as_3d = false;
    }

    // Check if we need to slide the matrix B
    const unsigned int num_dimensions_input0 = _input0->info()->num_dimensions();
    _slide_matrix_b                          = (_input1->info()->num_dimensions() >= num_dimensions_input0);

    ElementsProcessed num_elements_processed{};

    // Configure kernel window
    auto win_config = validate_and_configure_window(input0->info(), input1->info(), input2 != nullptr ? input2->info() : nullptr, output->info(), lhs_info, rhs_info, gemm_info, num_elements_processed);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

    // If _reinterpret_input_as_3d = _reinterpret_output_as_3d = true,
    // we will dispatch a batched-GEMM to reduce the complexity of the address calculation within the OpenCL kernel.
    // This means that the actual m used by the kernel is given by output->info()->dimension(1) and not by gemm_info.m
    const unsigned int internal_m = _reinterpret_output_as_3d ? gemm_info.m : output->info()->dimension(1);

    const unsigned int h_gemm_3d = _reinterpret_output_as_3d ? output->info()->dimension(1) : input0->info()->dimension(1);
    const unsigned int d_gemm_3d = _reinterpret_output_as_3d ? output->info()->dimension(2) : input0->info()->dimension(2);

    // Calculate partial (store instead of load) M0 and partial N0 for the partial blocks at the end of a row/column if any. This is to avoid padding.
    const unsigned int partial_store_m0 = internal_m % lhs_info.m0;
    const unsigned int partial_store_n0 = gemm_info.n % rhs_info.n0;

    // Shrink M0 to be always <= M (internal_m) to prevent out-of-bounds reads.
    // NOTE: This might have implications on heuristics and performance
    const unsigned int internal_m0 = std::min(internal_m, lhs_info.m0);

    // Create build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input0->info()->data_type()));
    build_opts.add_option_if(!(helpers::float_ops::is_one(alpha)), "-DALPHA=" + float_to_string_with_full_precision(alpha));
    build_opts.add_option_if(_input2 != nullptr, "-DBETA=" + float_to_string_with_full_precision(beta));
    build_opts.add_option_if(helpers::float_ops::is_one(beta), "-DUNIT_BETA");
    build_opts.add_option_if(_reinterpret_input_as_3d, "-DREINTERPRET_INPUT_AS_3D");
    build_opts.add_option_if(_reinterpret_output_as_3d, "-DREINTERPRET_OUTPUT_AS_3D");
    build_opts.add_option_if(gemm_info.broadcast_bias, "-DBROADCAST_BIAS");
    build_opts.add_option_if(_reinterpret_input_as_3d || _reinterpret_output_as_3d, "-DHEIGHT_GEMM3D=" + support::cpp11::to_string(h_gemm_3d));
    build_opts.add_option_if(_reinterpret_input_as_3d || _reinterpret_output_as_3d, "-DDEPTH_GEMM3D=" + support::cpp11::to_string(d_gemm_3d));
    build_opts.add_option_if(!_slide_matrix_b, "-DMATRIX_B_DEPTH=" + support::cpp11::to_string(input1->info()->dimension(2)));
    build_opts.add_option_if(rhs_info.interleave, "-DRHS_INTERLEAVE");
    build_opts.add_option_if(_use_dummy_work_items, "-DDUMMY_WORK_ITEMS");
    build_opts.add_option_if(rhs_info.export_to_cl_image, "-DOPENCL_IMAGE_SUPPORT");
    build_opts.add_option("-DRHS_HEIGHT=" + support::cpp11::to_string(input1->info()->dimension(1)));
    build_opts.add_option("-DM=" + support::cpp11::to_string(internal_m));
    build_opts.add_option("-DN=" + support::cpp11::to_string(gemm_info.n));
    build_opts.add_option("-DK=" + support::cpp11::to_string(gemm_info.k));
    build_opts.add_option("-DM0=" + support::cpp11::to_string(internal_m0));
    build_opts.add_option("-DN0=" + support::cpp11::to_string(rhs_info.n0));
    build_opts.add_option("-DK0=" + support::cpp11::to_string(rhs_info.k0));
    build_opts.add_option("-DH0=" + support::cpp11::to_string(rhs_info.h0));
    build_opts.add_option("-DPARTIAL_STORE_M0=" + support::cpp11::to_string(partial_store_m0));
    build_opts.add_option("-DPARTIAL_STORE_N0=" + support::cpp11::to_string(partial_store_n0));
    build_opts.add_option_if(gemm_info.activation_info.enabled(), "-DACTIVATION_TYPE=" + lower_string(string_from_activation_func(gemm_info.activation_info.activation())));
    build_opts.add_option_if(gemm_info.activation_info.enabled(), "-DA_VAL=" + float_to_string_with_full_precision(gemm_info.activation_info.a()));
    build_opts.add_option_if(gemm_info.activation_info.enabled(), "-DB_VAL=" + float_to_string_with_full_precision(gemm_info.activation_info.b()));

    std::string kernel_name("gemm_mm_reshaped_only_rhs_");
    kernel_name += rhs_info.transpose ? "t" : "nt";
    kernel_name += rhs_info.export_to_cl_image ? "_texture" : "";

    // Create kernel
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += (_add_bias ? "add_bias_" : "");
    _config_id += (_broadcast_bias ? "broadcast_bias_" : "");
    _config_id += (_reinterpret_input_as_3d ? "3di_" : "");
    _config_id += (_reinterpret_output_as_3d ? "3do_" : "");
    _config_id += (gemm_info.activation_info.enabled() ? "fused_activation_" : "");
    _config_id += lower_string(string_from_data_type(input0->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(gemm_info.k);
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(2));
    _config_id += "_";
    _config_id += support::cpp11::to_string(lhs_info.m0);
    _config_id += "_";
    _config_id += support::cpp11::to_string(rhs_info.n0);
    _config_id += "_";
    _config_id += support::cpp11::to_string(rhs_info.k0);
    _config_id += "_";
    _config_id += support::cpp11::to_string(rhs_info.h0);
    _config_id += "_";
    _config_id += support::cpp11::to_string(rhs_info.interleave);
}

Status CLGEMMMatrixMultiplyReshapedOnlyRHSKernel::validate(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, float alpha, float beta,
                                                           const GEMMLHSMatrixInfo &lhs_info,
                                                           const GEMMRHSMatrixInfo &rhs_info, const GEMMKernelInfo &gemm_info)
{
    ElementsProcessed num_elements_processed{};
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input0, input1, input2, output, alpha, beta, lhs_info, rhs_info, gemm_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input0->clone().get(),
                                                              input1->clone().get(),
                                                              input2 != nullptr ? input2->clone().get() : nullptr,
                                                              output->clone().get(),
                                                              lhs_info,
                                                              rhs_info,
                                                              gemm_info,
                                                              num_elements_processed)
                                .first);

    return Status{};
}

void CLGEMMMatrixMultiplyReshapedOnlyRHSKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    if(_input1->info()->num_dimensions() < 3)
    {
        // The stride_z for matrix B must be zero if we do not slice
        ARM_COMPUTE_ERROR_ON(_input1->info()->strides_in_bytes()[3] != 0);
    }

    Window slice          = window.first_slice_window_3D();
    Window slice_matrix_b = slice;

    slice_matrix_b.set(Window::DimX, Window::Dimension(0, 1, 1));
    slice_matrix_b.set(Window::DimY, Window::Dimension(0, 1, 1));

    const unsigned int total_cross_plane_pad_lhs = _input0->info()->padding().top + _input0->info()->padding().bottom;
    const unsigned int total_cross_plane_pad_out = _output->info()->padding().top + _output->info()->padding().bottom;

    cl::Image2D input1_image2d;

    if(_export_to_cl_image)
    {
        const TensorShape shape2d(_input1->info()->dimension(0) / 4, _input1->info()->dimension(1) * _input1->info()->dimension(2));
        const size_t      image_row_pitch = _input1->info()->strides_in_bytes()[1];

        input1_image2d = create_image2d_from_buffer(CLKernelLibrary::get().context(), _input1->cl_buffer(), shape2d, CL_FLOAT, image_row_pitch);
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
        add_2D_tensor_argument(idx, _input0, slice);

        // RHS buffer or RHS OpenCL image (_export_to_cl_image == true)
        if(_export_to_cl_image)
        {
            _kernel.setArg(idx++, input1_image2d);
        }
        else
        {
            add_2D_tensor_argument(idx, _input1, slice_b);
        }

        // Bias buffer (_add_bias == true)
        add_2D_tensor_argument_if(_add_bias, idx, _input2, slice);

        // Output buffer
        add_2D_tensor_argument(idx, _output, slice);

        // LHS stride_z
        _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(_input0->info()->strides_in_bytes()[2]));

        // RHS stride_z (not used if _export_to_cl_image == true)
        _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(_input1->info()->strides_in_bytes()[2]));

        // Bias stride_z (if _add_bias == true)
        if(_add_bias)
        {
            _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(_input2->info()->strides_in_bytes()[2]));
        }

        // Output stride_z
        _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(_output->info()->strides_in_bytes()[2]));

        // Cross-plan padding (if _reinterpret_input_as_3d = true)
        if(_reinterpret_input_as_3d)
        {
            _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(total_cross_plane_pad_lhs));
        }

        // Cross-plan padding (if _reinterpret_output_as_3d = true)
        if(_reinterpret_output_as_3d)
        {
            _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(total_cross_plane_pad_out));
        }

        enqueue(queue, *this, slice, lws_hint(), _use_dummy_work_items);
    }
    while(window.slide_window_slice_3D(slice));
}
} // namespace arm_compute
