/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#include "src/gpu/cl/kernels/ClGemmLowpQuantizeDownInt32ScaleByFixedPointKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"

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
Status validate_arguments(const ITensorInfo *src, const ITensorInfo *bias, const ITensorInfo *dst, const GEMMLowpOutputStageInfo *info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::S32);

    // Check biases if exist
    if(bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, bias);
        ARM_COMPUTE_RETURN_ERROR_ON(bias->num_dimensions() > 1);
        ARM_COMPUTE_RETURN_ERROR_ON(src->dimension(0) != bias->dimension(0));
    }

    if(dst->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(dst->data_type() != info->output_data_type, "Mismatching dst data type");
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, dst);
    }

    return Status{};
}
} // namespace

ClGemmLowpQuantizeDownInt32ScaleByFixedPointKernel::ClGemmLowpQuantizeDownInt32ScaleByFixedPointKernel()
{
    _type = CLKernelType::ELEMENTWISE;
}

Status ClGemmLowpQuantizeDownInt32ScaleByFixedPointKernel::validate(const ITensorInfo *src, const ITensorInfo *bias, const ITensorInfo *dst,
                                                                    const GEMMLowpOutputStageInfo *info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, bias, dst, info));

    return Status{};
}

void ClGemmLowpQuantizeDownInt32ScaleByFixedPointKernel::configure(const CLCompileContext &compile_context, const ITensorInfo *src, const ITensorInfo *bias, ITensorInfo *dst,
                                                                   const GEMMLowpOutputStageInfo *info)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, bias, dst, info));

    auto padding_info = get_padding_info({ src, bias, dst });

    // dst auto inizialitation if not yet initialized
    auto_init_if_empty(*dst, src->clone()->set_data_type(info->output_data_type));

    const unsigned int num_elems_processed_per_iteration = adjust_vec_size(4, src->dimension(0));

    // Set the arguments to pass at compile time
    auto           min = info->gemmlowp_min_bound;
    auto           max = info->gemmlowp_max_bound;
    CLBuildOptions build_opts;
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration));
    build_opts.add_option("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(src->dimension(0) % num_elems_processed_per_iteration));
    build_opts.add_option("-DRESULT_OFFSET_AFTER_SHIFT=" + support::cpp11::to_string(info->gemmlowp_offset));
    build_opts.add_option("-DRESULT_FIXEDPOINT_MULTIPLIER=" + support::cpp11::to_string(info->gemmlowp_multiplier));
    build_opts.add_option("-DRESULT_SHIFT=" + support::cpp11::to_string(info->gemmlowp_shift));
    build_opts.add_option("-DOUTPUT_DATA_TYPE=" + get_cl_type_from_data_type(dst->data_type()));
    build_opts.add_option_if((min > std::get<0>(quantization::get_min_max_values_from_quantized_data_type(info->output_data_type))) && (min != max),
                             "-DMIN_BOUND=" + support::cpp11::to_string(min));
    build_opts.add_option_if((max < std::get<1>(quantization::get_min_max_values_from_quantized_data_type(info->output_data_type))) && (min != max),
                             "-DMAX_BOUND=" + support::cpp11::to_string(max));
    build_opts.add_option_if(bias != nullptr, "-DADD_BIAS");

    // Create kernel
    const std::string kernel_name = (info->output_data_type == DataType::QSYMM16) ? "gemmlowp_output_stage_quantize_down_fixedpoint_qsymm16" : "gemmlowp_output_stage_quantize_down_fixedpoint";
    _kernel                       = create_kernel(compile_context, kernel_name, build_opts.options());

    // Configure kernel window
    auto win = calculate_max_window(*dst, Steps(num_elems_processed_per_iteration));
    ICLKernel::configure_internal(win);

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

void ClGemmLowpQuantizeDownInt32ScaleByFixedPointKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const auto src  = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    const auto bias = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_BIAS));
    auto       dst  = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    // Create src window
    Window collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice     = collapsed.first_slice_window_3D();

    // Setup bias slice
    unsigned int idx1 = num_arguments_per_3D_tensor();
    if(bias != nullptr)
    {
        Window biases_slice(slice);
        biases_slice.set(Window::DimY, Window::Dimension(0, 1, 1));
        biases_slice.set(Window::DimZ, Window::Dimension(0, 1, 1));
        add_1D_tensor_argument(idx1, bias, biases_slice);
    }

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, src, slice);
        add_3D_tensor_argument(idx1, dst, slice);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(collapsed.slide_window_slice_3D(slice));
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
