/*
 * Copyright (c) 2018-2022 Arm Limited.
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
#include "src/gpu/cl/kernels/ClWidthConcatenateKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"
#include "src/core/CL/CLValidate.h"
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
Status validate_arguments(const ITensorInfo *src, unsigned int width_offset, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON(src->data_type() == DataType::UNKNOWN);

    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON(src->dimension(0) + width_offset > dst->dimension(0));

    for(size_t i = 1; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(src->dimension(i) != dst->dimension(i));
    }
    ARM_COMPUTE_RETURN_ERROR_ON(src->num_dimensions() > 4);

    return Status{};
}
} // namespace

ClWidthConcatenateKernel::ClWidthConcatenateKernel()
{
    _type = CLKernelType::ELEMENTWISE;
}

Status ClWidthConcatenateKernel::validate(const ITensorInfo *src, unsigned int width_offset, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, width_offset, dst));
    return Status{};
}

void ClWidthConcatenateKernel::configure(const CLCompileContext &compile_context, ITensorInfo *src, unsigned int width_offset, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, width_offset, dst));

    auto padding_info = get_padding_info({ src, dst });

    const unsigned int num_elems_processed_per_iteration = adjust_vec_size(16, src->dimension(0));

    // Add build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(src->data_type()));
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration));
    build_opts.add_option("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(src->dimension(0) % num_elems_processed_per_iteration));
    build_opts.add_option("-DWIDTH_OFFSET=" + support::cpp11::to_string(width_offset));

    if(is_data_type_quantized_asymmetric(src->data_type()) && src->quantization_info() != dst->quantization_info())
    {
        const UniformQuantizationInfo iqinfo = src->quantization_info().uniform();
        const UniformQuantizationInfo oqinfo = dst->quantization_info().uniform();

        build_opts.add_option("-DOFFSET_IN1=" + float_to_string_with_full_precision(iqinfo.offset));
        build_opts.add_option("-DOFFSET_OUT=" + float_to_string_with_full_precision(oqinfo.offset));
        build_opts.add_option("-DSCALE_IN1=" + float_to_string_with_full_precision(iqinfo.scale));
        build_opts.add_option("-DSCALE_OUT=" + float_to_string_with_full_precision(oqinfo.scale));
    }
    _depth                  = src->dimension(2);
    std::string kernel_name = "concatenate_width";

    // A macro guard to compile ONLY the kernel of interest
    build_opts.add_option("-D" + upper_string(kernel_name));

    // Create kernel
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Configure kernel window
    Window win = calculate_max_window(*src, Steps(num_elems_processed_per_iteration));
    ICLKernel::configure_internal(win.collapse(win, Window::DimZ));

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

void ClWidthConcatenateKernel::run_op(ITensorPack &tensors, const Window &window, ::cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const auto src = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    auto       dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    unsigned int idx = 0;
    add_4D_tensor_argument(idx, src, window);
    add_4D_tensor_argument(idx, dst, window);
    _kernel.setArg<cl_uint>(idx++, _depth);
    enqueue(queue, *this, window, lws_hint());
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
