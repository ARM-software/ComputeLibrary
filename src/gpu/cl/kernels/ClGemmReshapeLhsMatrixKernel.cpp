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
#include "src/gpu/cl/kernels/ClGemmReshapeLhsMatrixKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/StringUtils.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
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
Status validate_arguments(const ITensorInfo *src, const ITensorInfo *dst, const GEMMLHSMatrixInfo &lhs_info, bool reinterpret_input_as_3d)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON(lhs_info.m0 == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(lhs_info.k0 == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(lhs_info.v0 == 0);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(((lhs_info.k0 & (lhs_info.k0 - 1)) && lhs_info.k0 != 3), "Only 2,3,4,8,16 are supported for k0");
    ARM_COMPUTE_RETURN_ERROR_ON(lhs_info.k0 > 16);
    ARM_COMPUTE_RETURN_ERROR_ON(lhs_info.m0 < 2 || lhs_info.m0 > 8);
    ARM_COMPUTE_RETURN_ERROR_ON((lhs_info.m0 > 4 && lhs_info.m0 < 8) && lhs_info.transpose);

    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON(src->data_type() == DataType::UNKNOWN);

    if(dst->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(dst->tensor_shape(),
                                                           misc::shape_calculator::compute_lhs_reshaped_shape(*src, lhs_info, reinterpret_input_as_3d));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(src, dst);
    }

    return Status{};
}

Window configure_window(ITensorInfo *src, ITensorInfo *dst, const GEMMLHSMatrixInfo &lhs_info, bool reinterpret_input_as_3d)
{
    const unsigned int num_elems_processed_per_iteration_x = lhs_info.k0;
    const unsigned int num_elems_processed_per_iteration_y = lhs_info.m0;

    TensorInfo tmp_info(*src);

    if(reinterpret_input_as_3d)
    {
        // Since the src tensor has to be reinterpreted as 3D and the execute window is based on a 2D interleave,
        // the window needs to be constructed on the 2D collapsed version of the tensor
        TensorShape tmp_shape(src->tensor_shape());
        tmp_shape.collapse(2U, 1U);
        tmp_info.set_tensor_shape(tmp_shape);
    }

    // dst auto inizialitation if not yet initialized
    auto_init_if_empty(*dst, src->clone()->set_tensor_shape(misc::shape_calculator::compute_lhs_reshaped_shape(*src, lhs_info, reinterpret_input_as_3d)));

    // Configure window
    Window win = calculate_max_window(tmp_info, Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));

    // Collapse along the Z direction
    // This collapse needs to be here in order to tune the Z dimension of LWS
    Window collapsed = win.collapse(win, Window::DimZ);

    return collapsed;
}
} // namespace

ClGemmReshapeLhsMatrixKernel::ClGemmReshapeLhsMatrixKernel()
{
    _type = CLKernelType::ELEMENTWISE;
}

void ClGemmReshapeLhsMatrixKernel::configure(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *dst, const GEMMLHSMatrixInfo &lhs_info, bool reinterpret_input_as_3d)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    // Perform validate step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst, lhs_info, reinterpret_input_as_3d));

    auto padding_info = get_padding_info({ src });

    const unsigned int src_w      = src->dimension(0);
    const unsigned int m          = reinterpret_input_as_3d ? src->dimension(1) * src->dimension(2) : src->dimension(1);
    const unsigned int partial_m0 = m % lhs_info.m0;
    const unsigned int partial_k0 = src_w % lhs_info.k0;

    // Create build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DM0=" + support::cpp11::to_string(lhs_info.m0));
    build_opts.add_option("-DK0=" + support::cpp11::to_string(lhs_info.k0));
    build_opts.add_option_if(lhs_info.interleave, "-DINTERLEAVE");
    build_opts.add_option_if_else(lhs_info.transpose, "-DRESHAPE_LHS_T", "-DRESHAPE_LHS_NT");
    build_opts.add_option("-DDATA_TYPE=" + get_cl_unsigned_type_from_element_size(src->element_size()));
    build_opts.add_option("-DPARTIAL_M0=" + support::cpp11::to_string(partial_m0));
    build_opts.add_option("-DPARTIAL_K0=" + support::cpp11::to_string(partial_k0));

    std::string kernel_name("gemm_reshape_lhs_matrix_");
    kernel_name += lhs_info.transpose ? "t" : "nt";

    // Create kernel
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Configure kernel window
    auto win_config = configure_window(src, dst, lhs_info, reinterpret_input_as_3d);
    ICLKernel::configure_internal(win_config);

    unsigned int idx = 2 * num_arguments_per_3d_tensor_nhw();
    _kernel.setArg<cl_int>(idx++, m);
    _kernel.setArg<cl_int>(idx++, lhs_info.v0);

    // Set config_id for enabling LWS tuning
    _config_id = "gemm_reshape_lhs_matrix_";
    _config_id += (reinterpret_input_as_3d ? "3d_" : "");
    _config_id += lower_string(string_from_data_type(src->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(2));
    _config_id += "_";
    _config_id += support::cpp11::to_string(lhs_info.m0);
    _config_id += "_";
    _config_id += support::cpp11::to_string(lhs_info.k0);
    _config_id += "_";
    _config_id += support::cpp11::to_string(lhs_info.v0);
    _config_id += "_";
    _config_id += support::cpp11::to_string(lhs_info.interleave);
    _config_id += "_";
    _config_id += support::cpp11::to_string(lhs_info.transpose);

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status ClGemmReshapeLhsMatrixKernel::validate(const ITensorInfo *src, const ITensorInfo *dst, const GEMMLHSMatrixInfo &lhs_info, bool reinterpret_input_as_3d)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst, lhs_info, reinterpret_input_as_3d));
    return Status{};
}

void ClGemmReshapeLhsMatrixKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const auto src = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    auto       dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    Window slice = window.first_slice_window_3D();

    do
    {
        unsigned int idx = 0;
        add_3d_tensor_nhw_argument(idx, src);
        add_3d_tensor_nhw_argument(idx, dst);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(window.slide_window_slice_3D(slice));
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
