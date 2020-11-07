/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "src/core/CL/kernels/CLGEMMLowpReductionKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace
{
Status validate_arguments_matrix_a_reduction(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::QSYMM8);

    if(output->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::S32);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->dimension(0) != input->dimension(1), "Output vector must have length equal to the number of rows of the input matrix");
    }
    return Status{};
}

Status validate_arguments_matrix_b_reduction(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::QSYMM8, DataType::QSYMM8_PER_CHANNEL);

    if(output->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::S32);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->dimension(0) != input->dimension(0), "Output vector must have length equal to the number of columns of the input matrix");
    }
    return Status{};
}
} // namespace

ICLGEMMLowpReductionKernel::ICLGEMMLowpReductionKernel()
    : _input(), _output()
{
}

void CLGEMMLowpMatrixAReductionKernel::configure(const ICLTensor *mtx_a, ICLTensor *vector_sum_row, const GEMMLowpReductionKernelInfo &info)
{
    configure(CLKernelLibrary::get().get_compile_context(), mtx_a, vector_sum_row, info);
}

void CLGEMMLowpMatrixAReductionKernel::configure(const CLCompileContext &compile_context, const ICLTensor *mtx_a, ICLTensor *vector_sum_row, const GEMMLowpReductionKernelInfo &info)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(mtx_a, vector_sum_row);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_matrix_a_reduction(mtx_a->info(), vector_sum_row->info()));

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*vector_sum_row->info(), TensorShape(mtx_a->info()->dimension(1)), 1, DataType::S32);

    auto padding_info = get_padding_info({ mtx_a, vector_sum_row });

    _input  = mtx_a;
    _output = vector_sum_row;

    // Set the arguments to pass at compile time
    CLBuildOptions build_opts;
    build_opts.add_option("-DCOLS_A=" + support::cpp11::to_string(mtx_a->info()->dimension(0)));
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(mtx_a->info()->data_type()));
    build_opts.add_option("-DACC_DATA_TYPE=" + get_cl_dot8_acc_type_from_data_type(mtx_a->info()->data_type()));
    build_opts.add_option_if(info.mul_by_scalar, "-DSCALAR=" + support::cpp11::to_string(info.scalar));

    const bool is_dot8_supported = dot8_supported(CLKernelLibrary::get().get_device());

    std::string kernel_name = "gemmlowp_matrix_a_reduction" + std::string(is_dot8_supported ? "_dot8" : "");

    // Create kernel
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Configure kernel window
    // This kernel does not need padding
    Window win = calculate_max_window(*vector_sum_row->info(), Steps());
    ICLKernel::configure_internal(win);

    _config_id = kernel_name;
    _config_id += "_";
    _config_id += support::cpp11::to_string(_input->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(_input->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(_input->info()->dimension(2));

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status CLGEMMLowpMatrixAReductionKernel::validate(const ITensorInfo *mtx_a, const ITensorInfo *vector_sum_row, const GEMMLowpReductionKernelInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_matrix_a_reduction(mtx_a, vector_sum_row));

    return Status{};
}

void CLGEMMLowpMatrixAReductionKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

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
        add_3D_tensor_argument(idx, _input, slice_in);
        add_2D_tensor_argument(idx, _output, slice_out);
        enqueue(queue, *this, slice_out, lws_hint());
    }
    while(collapsed.slide_window_slice_2D(slice_out));
}

void CLGEMMLowpMatrixBReductionKernel::configure(const ICLTensor *mtx_b, ICLTensor *vector_sum_col, const GEMMLowpReductionKernelInfo &info)
{
    configure(CLKernelLibrary::get().get_compile_context(), mtx_b, vector_sum_col, info);
}

void CLGEMMLowpMatrixBReductionKernel::configure(const CLCompileContext &compile_context, const ICLTensor *mtx_b, ICLTensor *vector_sum_col, const GEMMLowpReductionKernelInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(mtx_b, vector_sum_col);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_matrix_b_reduction(mtx_b->info(), vector_sum_col->info()));

    _input  = mtx_b;
    _output = vector_sum_col;

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*_output->info(), TensorShape(mtx_b->info()->dimension(0)), 1, DataType::S32);

    auto padding_info = get_padding_info({ mtx_b, vector_sum_col });

    const unsigned int num_elems_processed_per_iteration = adjust_vec_size(16, mtx_b->info()->dimension(0));

    // Set the arguments to pass at compile time
    CLBuildOptions build_opts;
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration));
    build_opts.add_option("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(mtx_b->info()->dimension(0) % num_elems_processed_per_iteration));
    build_opts.add_option("-DCOLS_B=" + support::cpp11::to_string(mtx_b->info()->dimension(0)));
    build_opts.add_option("-DROWS_B=" + support::cpp11::to_string(mtx_b->info()->dimension(1)));
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(mtx_b->info()->data_type()));
    build_opts.add_option("-DACC_DATA_TYPE=" + get_cl_dot8_acc_type_from_data_type(mtx_b->info()->data_type()));
    build_opts.add_option_if(info.mul_by_scalar, "-DSCALAR=" + support::cpp11::to_string(info.scalar));

    // Create kernel
    _kernel = create_kernel(compile_context, "gemmlowp_matrix_b_reduction", build_opts.options());

    // Configure kernel window
    Window win = calculate_max_window(*_output->info(), Steps(num_elems_processed_per_iteration));
    ICLKernel::configure_internal(win);

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status CLGEMMLowpMatrixBReductionKernel::validate(const ITensorInfo *mtx_b, const ITensorInfo *vector_sum_col, const GEMMLowpReductionKernelInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_matrix_b_reduction(mtx_b, vector_sum_col));

    return Status{};
}

void CLGEMMLowpMatrixBReductionKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window collapsed = window.collapse_if_possible(IKernel::window(), Window::DimY);

    Window slice_out = collapsed.first_slice_window_2D();
    Window slice_in  = slice_out;

    slice_in.set(Window::DimY, Window::Dimension(0, 0, 0));
    slice_in.set(Window::DimZ, Window::Dimension(0, 0, 0));

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice_in);
        add_2D_tensor_argument(idx, _output, slice_out);
        enqueue(queue, *this, slice_out, lws_hint());
    }
    while(collapsed.slide_window_slice_2D(slice_out));
}
} // namespace arm_compute
