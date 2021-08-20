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
#include "src/core/gpu/cl/kernels/ClGemmLowpQuantizeDownInt32ScaleKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"
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
Status validate_arguments(const ITensorInfo *src, const ITensorInfo *bias, const ITensorInfo *dst, const GEMMLowpOutputStageInfo *output_stage)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::S32);
    ARM_COMPUTE_RETURN_ERROR_ON((output_stage->output_data_type != DataType::QASYMM8) && (output_stage->output_data_type != DataType::QASYMM8_SIGNED));
    ARM_COMPUTE_RETURN_ERROR_ON(output_stage->gemmlowp_max_bound > std::get<1>(quantization::get_min_max_values_from_quantized_data_type(output_stage->output_data_type)));
    ARM_COMPUTE_RETURN_ERROR_ON(output_stage->gemmlowp_min_bound < std::get<0>(quantization::get_min_max_values_from_quantized_data_type(output_stage->output_data_type))
                                || output_stage->gemmlowp_min_bound > output_stage->gemmlowp_max_bound);

    // Check biases if exist
    if(bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, bias);
        ARM_COMPUTE_RETURN_ERROR_ON(bias->num_dimensions() > 1);
        ARM_COMPUTE_RETURN_ERROR_ON(src->dimension(0) != bias->dimension(0));
    }

    if(dst->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(dst->data_type() != output_stage->output_data_type, "Mismatching output data type");
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, dst);
    }

    return Status{};
}
} //namespace

ClGemmLowpQuantizeDownInt32ScaleKernel::ClGemmLowpQuantizeDownInt32ScaleKernel()
{
    _type = CLKernelType::ELEMENTWISE;
}

Status ClGemmLowpQuantizeDownInt32ScaleKernel::validate(const ITensorInfo *src, const ITensorInfo *bias, const ITensorInfo *dst, const GEMMLowpOutputStageInfo *output_stage)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, bias, dst, output_stage));

    return Status{};
}

void ClGemmLowpQuantizeDownInt32ScaleKernel::configure(const CLCompileContext &compile_context, const ITensorInfo *src, const ITensorInfo *bias, ITensorInfo *dst,
                                                       const GEMMLowpOutputStageInfo *output_stage)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, bias, dst, output_stage));

    auto padding_info = get_padding_info({ src, bias, dst });

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*dst, src->clone()->set_data_type(output_stage->output_data_type));

    const unsigned int num_elems_processed_per_iteration = adjust_vec_size(4, src->dimension(0));

    // Set the arguments to pass at compile time
    auto           min = output_stage->gemmlowp_min_bound;
    auto           max = output_stage->gemmlowp_max_bound;
    CLBuildOptions build_opts;
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration));
    build_opts.add_option("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(src->dimension(0) % num_elems_processed_per_iteration));
    build_opts.add_option("-DRESULT_OFFSET=" + support::cpp11::to_string(output_stage->gemmlowp_offset));
    build_opts.add_option("-DRESULT_MULT_INT=" + support::cpp11::to_string(output_stage->gemmlowp_multiplier));
    build_opts.add_option("-DRESULT_SHIFT=" + support::cpp11::to_string(output_stage->gemmlowp_shift));
    build_opts.add_option_if((min > std::get<0>(quantization::get_min_max_values_from_quantized_data_type(output_stage->output_data_type))) && (min != max),
                             "-DMIN_BOUND=" + support::cpp11::to_string(min));
    build_opts.add_option_if((max < std::get<1>(quantization::get_min_max_values_from_quantized_data_type(output_stage->output_data_type))) && (min != max),
                             "-DMAX_BOUND=" + support::cpp11::to_string(max));
    build_opts.add_option("-DOUTPUT_DATA_TYPE=" + get_cl_type_from_data_type(dst->data_type()));
    build_opts.add_option_if(bias != nullptr, "-DADD_BIAS");

    // Create kernel
    _kernel = create_kernel(compile_context, "gemmlowp_output_stage_quantize_down", build_opts.options());

    // Configure kernel window
    Window win = calculate_max_window(*src, Steps(num_elems_processed_per_iteration));
    ICLKernel::configure_internal(win);

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

void ClGemmLowpQuantizeDownInt32ScaleKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const auto src  = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    const auto bias = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_BIAS));
    auto       dst  = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    Window collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice     = collapsed.first_slice_window_3D();

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