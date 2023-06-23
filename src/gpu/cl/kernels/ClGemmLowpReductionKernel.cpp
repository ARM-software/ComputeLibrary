/*
 * Copyright (c) 2017-2023 Arm Limited.
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
#include "src/gpu/cl/kernels/ClGemmLowpReductionKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/helpers/AdjustVecSize.h"
#include "arm_compute/core/utils/StringUtils.h"

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
Status validate_arguments_matrix_a_reduction(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::QSYMM8);

    if(dst->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, DataType::S32);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(dst->dimension(0) != src->dimension(1), "Output vector must have length equal to the number of rows of the input matrix");
    }
    return Status{};
}

Status validate_arguments_matrix_b_reduction(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::QSYMM8, DataType::QSYMM8_PER_CHANNEL);

    if(dst->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, DataType::S32);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(dst->dimension(0) != src->dimension(0), "Output vector must have length equal to the number of columns of the input matrix");
    }
    return Status{};
}
} // namespace

IClGemmLowpReductionKernel::IClGemmLowpReductionKernel()
{
    _type = CLKernelType::ELEMENTWISE;
}

void ClGemmLowpMatrixAReductionKernel::configure(const CLCompileContext &compile_context, const ITensorInfo *mtx_a, ITensorInfo *vector_sum_row, const GEMMLowpReductionKernelInfo &info)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(mtx_a, vector_sum_row);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_matrix_a_reduction(mtx_a, vector_sum_row));

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*vector_sum_row, TensorShape(mtx_a->dimension(1)), 1, DataType::S32);

    auto padding_info = get_padding_info({ mtx_a, vector_sum_row });

    // Set the arguments to pass at compile time
    CLBuildOptions build_opts;
    build_opts.add_option("-DCOLS_A=" + support::cpp11::to_string(mtx_a->dimension(0)));
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(mtx_a->data_type()));
    build_opts.add_option("-DACC_DATA_TYPE=" + get_cl_dot8_acc_type_from_data_type(mtx_a->data_type()));
    build_opts.add_option_if(info.mul_by_scalar, "-DSCALAR=" + support::cpp11::to_string(info.scalar));

    const bool is_dot8_supported = dot8_supported(CLKernelLibrary::get().get_device());

    std::string kernel_name = "gemmlowp_matrix_a_reduction" + std::string(is_dot8_supported ? "_dot8" : "");

    // A macro guard to compile ONLY the kernel of interest
    build_opts.add_option("-D" + upper_string(kernel_name));

    // Create kernel
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Configure kernel window
    // This kernel does not need padding
    Window win = calculate_max_window(*vector_sum_row, Steps());
    ICLKernel::configure_internal(win);

    _config_id = kernel_name;
    _config_id += "_";
    _config_id += support::cpp11::to_string(mtx_a->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(mtx_a->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(mtx_a->dimension(2));

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status ClGemmLowpMatrixAReductionKernel::validate(const ITensorInfo *mtx_a, const ITensorInfo *vector_sum_row, const GEMMLowpReductionKernelInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_matrix_a_reduction(mtx_a, vector_sum_row));

    return Status{};
}

void ClGemmLowpMatrixAReductionKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const auto src = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    auto       dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    Window collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimY);
    Window slice_in  = collapsed.first_slice_window_2D();
    Window slice_out = collapsed.first_slice_window_2D();

    // Setup input slice. Its dimensions are increased in the cl kernel.
    slice_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    slice_in.set(Window::DimY, Window::Dimension(0, 0, 0));
    slice_in.set(Window::DimZ, Window::Dimension(0, 0, 0));

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, src, slice_in);
        add_2D_tensor_argument(idx, dst, slice_out);
        enqueue(queue, *this, slice_out, lws_hint());
    }
    while(collapsed.slide_window_slice_2D(slice_out));
}

void ClGemmLowpMatrixBReductionKernel::configure(const CLCompileContext &compile_context, const ITensorInfo *mtx_b, ITensorInfo *vector_sum_col, const GEMMLowpReductionKernelInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(mtx_b, vector_sum_col);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_matrix_b_reduction(mtx_b, vector_sum_col));

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*vector_sum_col, TensorShape(mtx_b->dimension(0)), 1, DataType::S32);

    auto padding_info = get_padding_info({ mtx_b, vector_sum_col });

    const unsigned int num_elems_processed_per_iteration = adjust_vec_size(16, mtx_b->dimension(0));

    // Set the arguments to pass at compile time
    CLBuildOptions build_opts;
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration));
    build_opts.add_option("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(mtx_b->dimension(0) % num_elems_processed_per_iteration));
    build_opts.add_option("-DCOLS_B=" + support::cpp11::to_string(mtx_b->dimension(0)));
    build_opts.add_option("-DROWS_B=" + support::cpp11::to_string(mtx_b->dimension(1)));
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(mtx_b->data_type()));
    build_opts.add_option("-DACC_DATA_TYPE=" + get_cl_dot8_acc_type_from_data_type(mtx_b->data_type()));
    build_opts.add_option_if(info.mul_by_scalar, "-DSCALAR=" + support::cpp11::to_string(info.scalar));

    const std::string kernel_name = "gemmlowp_matrix_b_reduction";

    // A macro guard to compile ONLY the kernel of interest
    build_opts.add_option("-D" + upper_string(kernel_name));

    // Create kernel
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Configure kernel window
    Window win = calculate_max_window(*vector_sum_col, Steps(num_elems_processed_per_iteration));
    IClKernel::configure_internal(win);

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status ClGemmLowpMatrixBReductionKernel::validate(const ITensorInfo *mtx_b, const ITensorInfo *vector_sum_col, const GEMMLowpReductionKernelInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_matrix_b_reduction(mtx_b, vector_sum_col));

    return Status{};
}

void ClGemmLowpMatrixBReductionKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const auto src = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    auto       dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    Window collapsed = window.collapse_if_possible(IKernel::window(), Window::DimY);

    Window slice_out = collapsed.first_slice_window_2D();
    Window slice_in  = slice_out;

    slice_in.set(Window::DimY, Window::Dimension(0, 0, 0));
    slice_in.set(Window::DimZ, Window::Dimension(0, 0, 0));

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, src, slice_in);
        add_2D_tensor_argument(idx, dst, slice_out);
        enqueue(queue, *this, slice_out, lws_hint());
    }
    while(collapsed.slide_window_slice_2D(slice_out));
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
