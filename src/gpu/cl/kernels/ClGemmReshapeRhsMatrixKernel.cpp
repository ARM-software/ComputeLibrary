/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "src/gpu/cl/kernels/ClGemmReshapeRhsMatrixKernel.h"

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
Status validate_arguments(const ITensorInfo *src, const ITensorInfo *dst, const GEMMRHSMatrixInfo &rhs_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON(rhs_info.n0 == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(rhs_info.k0 == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(rhs_info.h0 == 0);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(((rhs_info.n0 & (rhs_info.n0 - 1)) && rhs_info.n0 != 3), "Only 2,3,4,8,16 are supported for n0");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(((rhs_info.k0 & (rhs_info.k0 - 1)) && (rhs_info.k0 != 1) && (rhs_info.k0 != 3)), "Only 1,2,3,4,8,16 are supported for k0");
    ARM_COMPUTE_RETURN_ERROR_ON(rhs_info.n0 > 16);
    ARM_COMPUTE_RETURN_ERROR_ON(rhs_info.k0 > 16);
    ARM_COMPUTE_RETURN_ERROR_ON((rhs_info.k0 == 1) && (rhs_info.transpose));

    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON(src->data_type() == DataType::UNKNOWN);

    if(rhs_info.export_to_cl_image)
    {
        const TensorInfo tensor_reshaped_info(misc::shape_calculator::compute_rhs_reshaped_shape(*src, rhs_info), 1, src->data_type());
        ARM_COMPUTE_RETURN_ON_ERROR(gemm::validate_image2d_support_on_rhs(tensor_reshaped_info, rhs_info));
    }

    if(dst->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(dst->tensor_shape(), misc::shape_calculator::compute_rhs_reshaped_shape(*src, rhs_info));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(src, dst);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *src, ITensorInfo *dst, const GEMMRHSMatrixInfo &rhs_info)
{
    const unsigned int num_elems_processed_per_iteration_x = rhs_info.n0;
    const unsigned int num_elems_processed_per_iteration_y = rhs_info.k0;
    bool               window_changed                      = false;

    // dst auto initialization if not yet initialized
    auto_init_if_empty(*dst, src->clone()->set_tensor_shape(misc::shape_calculator::compute_rhs_reshaped_shape(*src, rhs_info)));

    // Configure window
    Window win = calculate_max_window(*src, Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));

    AccessWindowRectangle src_access(src, 0, 0, num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y);

    window_changed = update_window_and_padding(win, src_access);

    if(rhs_info.export_to_cl_image)
    {
        gemm::update_padding_for_cl_image(dst);
    }

    // Collapse along the Z direction
    // This collapse needs to be here in order to tune the Z dimension of LWS
    Window collapsed = win.collapse(win, Window::DimZ);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, collapsed);
}
} // namespace

ClGemmReshapeRhsMatrixKernel::ClGemmReshapeRhsMatrixKernel()
{
    _type = CLKernelType::ELEMENTWISE;
}

void ClGemmReshapeRhsMatrixKernel::configure(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *dst, const GEMMRHSMatrixInfo &rhs_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    // Perform validate step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst, rhs_info));

    // Create build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DN0=" + support::cpp11::to_string(rhs_info.n0));
    build_opts.add_option("-DK0=" + support::cpp11::to_string(rhs_info.k0));
    build_opts.add_option_if(rhs_info.interleave, "-DINTERLEAVE");
    build_opts.add_option_if(rhs_info.transpose, "-DRESHAPE_RHS_T");
    build_opts.add_option_if(!rhs_info.transpose, "-DRESHAPE_RHS_NT");
    build_opts.add_option("-DDATA_TYPE=" + get_cl_unsigned_type_from_element_size(src->element_size()));

    std::string kernel_name("gemm_reshape_rhs_matrix_");
    kernel_name += rhs_info.transpose ? "t" : "nt";

    // Create kernel
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Configure kernel window
    auto win_config = validate_and_configure_window(src, dst, rhs_info);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

    unsigned int idx = 2 * num_arguments_per_3d_tensor_nhw();
    _kernel.setArg<cl_int>(idx++, rhs_info.h0);
}

Status ClGemmReshapeRhsMatrixKernel::validate(const ITensorInfo *src, const ITensorInfo *dst, const GEMMRHSMatrixInfo &rhs_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst, rhs_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(src->clone().get(), dst->clone().get(), rhs_info).first);

    return Status{};
}

void ClGemmReshapeRhsMatrixKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
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