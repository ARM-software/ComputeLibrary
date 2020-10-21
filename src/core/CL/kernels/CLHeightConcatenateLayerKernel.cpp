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
#include "src/core/CL/kernels/CLHeightConcatenateLayerKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/Cast.h"

#include "support/StringSupport.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, unsigned int height_offset, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(Window::DimY) + height_offset > output->dimension(Window::DimY));

    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(0) != output->dimension(0));
    for(size_t i = 2; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(i) != output->dimension(i));
    }
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 4);

    return Status{};
}
} // namespace

CLHeightConcatenateLayerKernel::CLHeightConcatenateLayerKernel()
    : _height_offset(0)
{
}

Status CLHeightConcatenateLayerKernel::validate(const ITensorInfo *input, unsigned int height_offset, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, height_offset, output));
    return Status{};
}

void CLHeightConcatenateLayerKernel::configure(const CLCompileContext &compile_context, ITensorInfo *input, unsigned int height_offset, ITensorInfo *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input, height_offset, output));

    auto padding_info = get_padding_info({ input, output });

    _height_offset = height_offset;

    // Add build options
    const unsigned int num_elems_processed_per_iteration = adjust_vec_size(4, input->dimension(0));

    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_unsigned_type_from_element_size(input->element_size()));
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration));
    build_opts.add_option("-DHEIGHT_OFFSET=" + support::cpp11::to_string(_height_offset));
    build_opts.add_option("-DDEPTH=" + support::cpp11::to_string(input->dimension(2)));
    build_opts.add_option("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(input->dimension(0) % num_elems_processed_per_iteration));

    if(is_data_type_quantized_asymmetric(input->data_type()) && input->quantization_info() != output->quantization_info())
    {
        const UniformQuantizationInfo iq_info = input->quantization_info().uniform();
        const UniformQuantizationInfo oq_info = output->quantization_info().uniform();

        build_opts.add_option("-DOFFSET_IN1=" + float_to_string_with_full_precision(iq_info.offset));
        build_opts.add_option("-DOFFSET_OUT=" + float_to_string_with_full_precision(oq_info.offset));
        build_opts.add_option("-DSCALE_IN1=" + float_to_string_with_full_precision(iq_info.scale));
        build_opts.add_option("-DSCALE_OUT=" + float_to_string_with_full_precision(oq_info.scale));
    }

    // Create kernel
    _kernel = create_kernel(compile_context, "concatenate_height", build_opts.options());
    // Configure kernel window

    // The window needs to be based on input as we copy all the heights of input
    Window win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
    ICLKernel::configure_internal(win.collapse(win, Window::DimZ));

    // Set output valid region
    output->set_valid_region(ValidRegion(Coordinates(), output->tensor_shape()));

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

void CLHeightConcatenateLayerKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const auto src = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    auto       dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    unsigned int idx = 0;
    add_4D_tensor_argument(idx, src, window);
    add_4D_tensor_argument(idx, dst, window);
    enqueue(queue, *this, window, lws_hint());
}
} // namespace arm_compute
