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
#include "src/gpu/cl/kernels/ClQuantizeKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/StringUtils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"

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
Status validate_arguments(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F32, DataType::F16);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src);

    // Output must always be initialized
    ARM_COMPUTE_RETURN_ERROR_ON(dst->tensor_shape().total_size() == 0);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::QASYMM16);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, dst);

    return Status{};
}
} // namespace

ClQuantizeKernel::ClQuantizeKernel()
{
    _type = CLKernelType::ELEMENTWISE;
}

void ClQuantizeKernel::configure(const CLCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    auto padding_info = get_padding_info({ src, dst });

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst));

    const int  vec_size_x     = 16 / src->element_size();
    const int  input_width_x  = src->tensor_shape().x();
    const bool multi_access_x = (input_width_x / vec_size_x > 0);

    const UniformQuantizationInfo qinfo            = dst->quantization_info().uniform();
    const DataType                output_data_type = dst->data_type();

    float   scale_to_apply  = qinfo.scale;
    int32_t offset_to_apply = qinfo.offset;
    if(is_data_type_quantized_asymmetric(src->data_type()))
    {
        /*
         * In case of requantization of a quantized input tensor to an output tensor with another quantization
         * instead of of apply dequantization and then a quantization functions, we just compute new scale and
         * offset to apply.
         *
         * Assuming:
         *   - q_i as input quantized value
         *   - q_o as output quantized value
         *   - z_i as input quantization offset value
         *   - z_o as output quantization offset value
         *   - s_i as input quantization scale value
         *   - s_o as output quantization scale value
         *   - z_n as new quantization offset value
         *   - s_n as new quantization scale value
         *
         * q_o = ( q_i - z_i ) * s_i / s_o + z_o
         *
         * We can rewrite the formula as:
         *
         * q_o = ( q_i * s_i / s_o ) - z_i * s_i / s_o + z_o
         *
         * q_o = q_i / s_n + z_n
         *
         * Where:
         *
         * s_n = s_o / s_i
         *
         * z_n = - z_i * s_i / s_o + z_o
         *
         */
        const UniformQuantizationInfo qinfo_in = src->quantization_info().uniform();
        scale_to_apply /= qinfo_in.scale;
        // In order to minimize flooring we convert the offset to a float,
        // then compute the new offset in the float domain,
        // finally we convert it back as int32_t
        offset_to_apply -= static_cast<int32_t>(static_cast<float>(qinfo_in.offset) * qinfo_in.scale / qinfo.scale);
    }

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option_if(is_data_type_float(src->data_type()), "-DIS_FLOAT");
    build_opts.add_option("-DSCALE=" + float_to_string_with_full_precision(scale_to_apply));
    build_opts.add_option("-DOFFSET=" + support::cpp11::to_string(offset_to_apply));
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(vec_size_x));
    build_opts.add_option("-DDATA_TYPE_IN=" + get_cl_type_from_data_type(src->data_type()));
    build_opts.add_option("-DDATA_TYPE_OUT=" + get_cl_type_from_data_type(output_data_type));
    build_opts.add_option_if(multi_access_x, "-DLAST_ACCESSED_X=" + support::cpp11::to_string(std::max<int>(input_width_x - vec_size_x, 0)));
    std::pair<int, int> min_max_quant_values = quantization::get_min_max_values_from_quantized_data_type(output_data_type);
    build_opts.add_option("-DMIN_QUANT_VAL=" + support::cpp11::to_string(min_max_quant_values.first));
    build_opts.add_option("-DMAX_QUANT_VAL=" + support::cpp11::to_string(min_max_quant_values.second));

    _kernel = create_kernel(compile_context, "quantization_layer", build_opts.options());

    // Configure kernel window
    Window win = calculate_max_window(*src, Steps());
    if(multi_access_x)
    {
        win.set(Window::DimX, Window::Dimension(win.x().start(), ceil_to_multiple(win.x().end(), vec_size_x), vec_size_x));
    }
    ICLKernel::configure_internal(win);

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status ClQuantizeKernel::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst));
    return Status{};
}

void ClQuantizeKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    auto src = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    auto dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), 3);
    Window slice            = window_collapsed.first_slice_window_3D();

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, src, slice);
        add_3D_tensor_argument(idx, dst, slice);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(window_collapsed.slide_window_slice_3D(slice));
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
