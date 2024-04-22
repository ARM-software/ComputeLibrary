/*
 * Copyright (c) 2017-2021, 2023 Arm Limited.
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
#include "src/gpu/cl/kernels/ClDequantizeKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/StringUtils.h"
#include "arm_compute/core/Validate.h"

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
Status validate_arguments(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED,
                                                         DataType::QSYMM8_PER_CHANNEL, DataType::QSYMM8,
                                                         DataType::QSYMM16);

    if (dst->tensor_shape().total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(dst);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, DataType::F16, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, dst);
    }

    return Status{};
}
} // namespace

ClDequantizeKernel::ClDequantizeKernel()
{
    _type = CLKernelType::ELEMENTWISE;
}

void ClDequantizeKernel::configure(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*dst, src->tensor_shape(), 1, DataType::F32);

    auto padding_info = get_padding_info({src, dst});

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst));

    const int  vec_size_x     = 16 / dst->element_size();
    const int  output_width_x = dst->tensor_shape().x();
    const bool multi_access_x = (output_width_x / vec_size_x > 0);

    const bool  is_quantized_per_channel = is_data_type_quantized_per_channel(src->data_type());
    std::string kernel_name              = "dequantization_layer";

    // Create kernel
    CLBuildOptions build_opts;
    if (!is_quantized_per_channel)
    {
        const UniformQuantizationInfo qinfo   = src->quantization_info().uniform();
        const int                     qoffset = is_data_type_quantized_asymmetric(src->data_type()) ? qinfo.offset : 0;
        build_opts.add_option("-DSCALE=" + float_to_string_with_full_precision(qinfo.scale));
        build_opts.add_option("-DOFFSET=" + support::cpp11::to_string(qoffset));
    }
    else
    {
        kernel_name += "_per_channel";
        kernel_name += src->data_layout() == DataLayout::NCHW ? "_nchw" : "_nhwc";
    }

    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(vec_size_x));
    build_opts.add_option("-DDATA_TYPE_SRC=" + get_cl_type_from_data_type(src->data_type()));
    build_opts.add_option("-DDATA_TYPE_DST=" + get_cl_type_from_data_type(dst->data_type()));
    build_opts.add_option_if(multi_access_x, "-DLAST_ACCESSED_X=" + support::cpp11::to_string(
                                                                        std::max<int>(output_width_x - vec_size_x, 0)));

    // Create kernel name
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Configure kernel window
    Window win = calculate_max_window(*dst);
    if (multi_access_x)
    {
        win.set(Window::DimX,
                Window::Dimension(win.x().start(), ceil_to_multiple(win.x().end(), vec_size_x), vec_size_x));
    }
    ICLKernel::configure_internal(win);

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status ClDequantizeKernel::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst));
    return Status{};
}

void ClDequantizeKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    auto src = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    auto dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    const bool is_quantized_per_channel = is_data_type_quantized_per_channel(src->info()->data_type());

    // Collapse windo
    Window new_window = is_quantized_per_channel ? window.collapse_if_possible(ICLKernel::window(), 4)
                                                 : window.collapse_if_possible(ICLKernel::window(), 3);
    Window slice      = new_window.first_slice_window_3D();

    if (is_quantized_per_channel)
    {
        unsigned int idx = num_arguments_per_3D_tensor() * 2; //Skip the input and output parameters
        _kernel.setArg(idx++, src->quantization().scale->cl_buffer());
    }

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, src, slice);
        add_3D_tensor_argument(idx, dst, slice);
        enqueue(queue, *this, slice, lws_hint());
    } while (new_window.slide_window_slice_3D(slice));
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
