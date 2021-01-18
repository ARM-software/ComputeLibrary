/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#include "src/core/gpu/cl/kernels/ClBatchConcatenateKernel.h"

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
Status validate_arguments(const ITensorInfo *src, unsigned int batch_offset, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON(src->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);

    ARM_COMPUTE_RETURN_ERROR_ON(src->dimension(Window::DimX) != dst->dimension(Window::DimX));
    ARM_COMPUTE_RETURN_ERROR_ON(src->dimension(Window::DimY) != dst->dimension(Window::DimY));
    ARM_COMPUTE_RETURN_ERROR_ON(src->dimension(Window::DimZ) != dst->dimension(Window::DimZ));
    ARM_COMPUTE_RETURN_ERROR_ON(src->dimension(3) + batch_offset > dst->dimension(3));
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(4, src, dst);

    return Status{};
}
} // namespace

ClBatchConcatenateKernel::ClBatchConcatenateKernel()
    : _batch_offset(0)
{
}

void ClBatchConcatenateKernel::configure(const CLCompileContext &compile_context, ITensorInfo *src, unsigned int batch_offset, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, batch_offset, dst));

    auto padding_info = get_padding_info({ src, dst });

    _batch_offset = batch_offset;

    const unsigned int num_elems_processed_per_iteration = adjust_vec_size(16 / src->element_size(), src->dimension(0));

    // Add build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(src->data_type()));
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration));
    build_opts.add_option("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(src->dimension(0) % num_elems_processed_per_iteration));
    if(is_data_type_quantized_asymmetric(src->data_type()) && src->quantization_info() != dst->quantization_info())
    {
        const UniformQuantizationInfo iq_info = src->quantization_info().uniform();
        const UniformQuantizationInfo oq_info = dst->quantization_info().uniform();

        build_opts.add_option("-DOFFSET_IN1=" + float_to_string_with_full_precision(iq_info.offset));
        build_opts.add_option("-DOFFSET_OUT=" + float_to_string_with_full_precision(oq_info.offset));
        build_opts.add_option("-DSCALE_IN1=" + float_to_string_with_full_precision(iq_info.scale));
        build_opts.add_option("-DSCALE_OUT=" + float_to_string_with_full_precision(oq_info.scale));
    }

    // Create kernel
    _kernel = create_kernel(compile_context, "concatenate", build_opts.options());

    // Configure kernel window
    auto win = calculate_max_window(*dst, Steps(num_elems_processed_per_iteration));
    win.set(3, Window::Dimension(0, src->tensor_shape()[3], 1));
    ICLKernel::configure_internal(win);

    // Set dst valid region
    dst->set_valid_region(ValidRegion(Coordinates(), dst->tensor_shape()));

    // Set config_id for enabling LWS tuning
    _config_id = "concatenate_";
    _config_id += support::cpp11::to_string(3);
    _config_id += "_";
    _config_id += support::cpp11::to_string(batch_offset);
    _config_id += "_";
    _config_id += support::cpp11::to_string(src->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(src->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(src->dimension(2));
    _config_id += "_";
    _config_id += support::cpp11::to_string(src->dimension(3));

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status ClBatchConcatenateKernel::validate(const arm_compute::ITensorInfo *src,
                                          unsigned int                    batch_offset,
                                          const arm_compute::ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, batch_offset, dst));
    return Status{};
}

void ClBatchConcatenateKernel::run_op(ITensorPack &tensors, const Window &window, ::cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const auto src = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    auto       dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    Window slice = window.first_slice_window_3D();

    const int offset_to_first_elements_in_bytes = _batch_offset * dst->info()->strides_in_bytes()[3];

    unsigned int idx = 2 * num_arguments_per_3D_tensor(); // Skip the src and dst parameters
    _kernel.setArg<cl_int>(idx, offset_to_first_elements_in_bytes);

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, src, slice);
        add_3D_tensor_argument(idx, dst, slice);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(window.slide_window_slice_3D(slice));
}
} // namespace opencl
} // namespace kernels
} // namespace arm_compute
