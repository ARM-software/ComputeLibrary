/*
 * Copyright (c) 2018-2023 Arm Limited.
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
#include "src/gpu/cl/kernels/ClWidthConcatenate4TensorsKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/helpers/AdjustVecSize.h"
#include "arm_compute/core/utils/StringUtils.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/core/utils/helpers/tensor_info.h"
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
Status validate_arguments(const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *src3, const ITensorInfo *src4, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src1, src2, src3, src4, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src1);
    ARM_COMPUTE_RETURN_ERROR_ON(src1->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src1, src2, src3, src4, dst);
    ARM_COMPUTE_RETURN_ERROR_ON(src1->dimension(0) + src2->dimension(0) + src3->dimension(0) + src4->dimension(0) > dst->dimension(0));

    for(size_t i = 1; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(src1->dimension(i) != dst->dimension(i));
        ARM_COMPUTE_RETURN_ERROR_ON(src2->dimension(i) != dst->dimension(i));
        ARM_COMPUTE_RETURN_ERROR_ON(src3->dimension(i) != dst->dimension(i));
        ARM_COMPUTE_RETURN_ERROR_ON(src4->dimension(i) != dst->dimension(i));
    }
    ARM_COMPUTE_RETURN_ERROR_ON(src1->num_dimensions() > 4);

    return Status{};
}
} // namespace

ClWidthConcatenate4TensorsKernel::ClWidthConcatenate4TensorsKernel()
{
    _type = CLKernelType::ELEMENTWISE;
}

Status ClWidthConcatenate4TensorsKernel::validate(const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *src3, const ITensorInfo *src4, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src1, src2, src3, src4, dst));
    return Status{};
}

void ClWidthConcatenate4TensorsKernel::configure(const CLCompileContext &compile_context,
                                                 ITensorInfo *src1, ITensorInfo *src2,
                                                 ITensorInfo *src3, ITensorInfo *src4,
                                                 ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src1, src2, src3, src4, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src1, src2, src3, src4, dst));

    auto               padding_info                      = get_padding_info({ src1, src2, src3, src4, dst });
    const unsigned int min_dimension                     = std::min(std::min(src1->dimension(0), src2->dimension(0)), std::min(src3->dimension(0), src4->dimension(0)));
    const unsigned int num_elems_processed_per_iteration = adjust_vec_size(8, min_dimension);
    const unsigned int vec_size_leftover                 = dst->dimension(0) % num_elems_processed_per_iteration;

    // Add build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(src1->data_type()));
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration));
    build_opts.add_option("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(vec_size_leftover));
    build_opts.add_option("-DELEMENT_SIZE=" + support::cpp11::to_string(src1->element_size()));
    build_opts.add_option("-DINPUT1_ROTATE_N=" + support::cpp11::to_string((src1->dimension(0) - vec_size_leftover) % num_elems_processed_per_iteration));
    build_opts.add_option("-DINPUT2_ROTATE_N=" + support::cpp11::to_string((src1->dimension(0) + src2->dimension(0) - vec_size_leftover) % num_elems_processed_per_iteration));
    build_opts.add_option("-DINPUT3_ROTATE_N=" + support::cpp11::to_string((src1->dimension(0) + src2->dimension(0) + src3->dimension(0) - vec_size_leftover) % num_elems_processed_per_iteration));

    _depth        = src1->dimension(2);
    _input1_width = src1->dimension(0);
    _input2_width = src2->dimension(0);
    _input3_width = src3->dimension(0);

    // If soources have different quantization info set quantization parameters needed for the re-quantization process
    const bool have_different_qinfo = helpers::tensor_info::tensors_have_different_quantization_info(dst, src1, src2, src3, src4);
    if(is_data_type_quantized_asymmetric(src1->data_type()) && have_different_qinfo)
    {
        const UniformQuantizationInfo iq1_info = src1->quantization_info().uniform();
        const UniformQuantizationInfo iq2_info = src2->quantization_info().uniform();
        const UniformQuantizationInfo iq3_info = src3->quantization_info().uniform();
        const UniformQuantizationInfo iq4_info = src4->quantization_info().uniform();
        const UniformQuantizationInfo oq_info  = dst->quantization_info().uniform();

        build_opts.add_option("-DOFFSET_IN1=" + float_to_string_with_full_precision(iq1_info.offset));
        build_opts.add_option("-DSCALE_IN1=" + float_to_string_with_full_precision(iq1_info.scale));
        build_opts.add_option("-DOFFSET_IN2=" + float_to_string_with_full_precision(iq2_info.offset));
        build_opts.add_option("-DSCALE_IN2=" + float_to_string_with_full_precision(iq2_info.scale));
        build_opts.add_option("-DOFFSET_IN3=" + float_to_string_with_full_precision(iq3_info.offset));
        build_opts.add_option("-DSCALE_IN3=" + float_to_string_with_full_precision(iq3_info.scale));
        build_opts.add_option("-DOFFSET_IN4=" + float_to_string_with_full_precision(iq4_info.offset));
        build_opts.add_option("-DSCALE_IN4=" + float_to_string_with_full_precision(iq4_info.scale));
        build_opts.add_option("-DOFFSET_OUT=" + float_to_string_with_full_precision(oq_info.offset));
        build_opts.add_option("-DSCALE_OUT=" + float_to_string_with_full_precision(oq_info.scale));
    }
    std::string kernel_name = "concatenate_width_x4";

    // A macro guard to compile ONLY the kernel of interest
    build_opts.add_option("-D" + upper_string(kernel_name));

    // Create kernel
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Configure kernel window
    Window win = calculate_max_window(*dst, Steps(num_elems_processed_per_iteration));
    ICLKernel::configure_internal(win.collapse(win, Window::DimZ));

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));

    // Set config_id for enabling LWS tuning
    _config_id = "concatenate_width_x4_";
    _config_id += lower_string(string_from_data_type(src1->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(src1->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(src1->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(src2->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(src2->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(src3->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(src3->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(src4->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(src4->dimension(1));
}

void ClWidthConcatenate4TensorsKernel::run_op(ITensorPack &tensors, const Window &window, ::cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const auto src0 = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_VEC));
    const auto src1 = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_VEC + 1));
    const auto src2 = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_VEC + 2));
    const auto src3 = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_VEC + 3));
    auto       dst  = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    Window slice = window.first_slice_window_4D();

    do
    {
        unsigned int idx = 0;
        add_4D_tensor_argument(idx, src0, slice);
        add_4D_tensor_argument(idx, src1, slice);
        add_4D_tensor_argument(idx, src2, slice);
        add_4D_tensor_argument(idx, src3, slice);
        add_4D_tensor_argument(idx, dst, slice);
        _kernel.setArg<cl_int>(idx++, _depth);
        _kernel.setArg<cl_int>(idx++, _input1_width);
        _kernel.setArg<cl_int>(idx++, _input2_width);
        _kernel.setArg<cl_int>(idx++, _input3_width);
        enqueue(queue, *this, window, lws_hint());
    }
    while(window.slide_window_slice_4D(slice));
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
