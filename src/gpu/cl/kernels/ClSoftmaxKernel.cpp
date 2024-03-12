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
#include "src/gpu/cl/kernels/ClSoftmaxKernel.h"

#include "arm_compute/core/CL/CLCompileContext.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/CoreTypes.h"
#include "arm_compute/core/Dimensions.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Steps.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/DataTypeUtils.h"
#include "arm_compute/core/utils/helpers/AdjustVecSize.h"
#include "arm_compute/core/utils/StringUtils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "src/core/helpers/WindowHelpers.h"
#include "support/Cast.h"
#include "support/StringSupport.h"

#include <string>

namespace arm_compute
{
namespace opencl
{
namespace kernels
{

ClSoftmaxKernel::ClSoftmaxKernel()
{
}

Status ClSoftmaxKernel::validate(const ITensorInfo &src, const ITensorInfo &dst, const SoftmaxKernelInfo &info)
{
    ARM_COMPUTE_UNUSED(src, dst, info);

    ARM_COMPUTE_RETURN_ERROR_ON(src.num_dimensions() > 4);

    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&src, &dst);

    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN( //
        &src, DataType::F32, DataType::F16, DataType::QASYMM8, DataType::QASYMM8_SIGNED);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src, &dst);

    ARM_COMPUTE_RETURN_ERROR_ON(info.input_data_type != src.data_type());
    ARM_COMPUTE_RETURN_ERROR_ON(info.axis < static_cast<int32_t>(-src.num_dimensions()) ||
                                static_cast<int32_t>(src.num_dimensions()) <= info.axis);

    if (is_data_type_quantized_asymmetric(src.data_type()))
    {
        ARM_COMPUTE_RETURN_ERROR_ON(src.quantization_info().uniform().scale < 0);

        ARM_COMPUTE_RETURN_ERROR_ON(dst.quantization_info() !=
                                    get_softmax_output_quantization_info(src.data_type(), info.is_log));
    }

    return Status{};
}

void ClSoftmaxKernel::configure(const CLCompileContext  &compile_context,
                                const ITensorInfo       &src,
                                ITensorInfo             &dst,
                                const SoftmaxKernelInfo &info)
{
    ARM_COMPUTE_UNUSED(compile_context, src, dst, info);

    const auto &dst_shape = dst.tensor_shape();

    const auto data_type    = src.data_type();
    const auto element_size = src.element_size();

    const auto is_quantized = data_type == DataType::QASYMM8 || data_type == DataType::QASYMM8_SIGNED;
    const auto src_qinfo    = src.quantization_info().uniform();
    const auto dst_qinfo    = dst.quantization_info().uniform();

    const auto axis   = wrap_around(info.axis, static_cast<int32_t>(src.num_dimensions()));
    const auto length = dst_shape[axis];

    const auto tmp_data_type = is_quantized ? DataType::F32 : data_type;

    const auto vec_size          = adjust_vec_size(16 / element_size, dst_shape[0]);
    const auto vec_size_leftover = dst_shape[0] % vec_size;

    std::string    kernel_name("softmax");
    CLBuildOptions build_opts;

    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type));
    build_opts.add_option("-DTMP_DATA_TYPE=" + get_cl_type_from_data_type(tmp_data_type));
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(vec_size));
    build_opts.add_option("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(vec_size_leftover));
    build_opts.add_option("-DLENGTH=" + support::cpp11::to_string(length));
    build_opts.add_option_if(info.is_log, "-DIS_LOG");
    build_opts.add_option("-DBETA=" + float_to_string_with_full_precision(info.beta));

    build_opts.add_option_if(is_quantized, "-DIS_QUANTIZED");
    build_opts.add_option_if(is_quantized, "-DSRC_OFFSET=" + float_to_string_with_full_precision(src_qinfo.offset));
    build_opts.add_option_if(is_quantized, "-DSRC_SCALE=" + float_to_string_with_full_precision(src_qinfo.scale));
    build_opts.add_option_if(is_quantized, "-DDST_OFFSET=" + float_to_string_with_full_precision(dst_qinfo.offset));
    build_opts.add_option_if(is_quantized, "-DDST_SCALE=" + float_to_string_with_full_precision(dst_qinfo.scale));

    if (axis == 0)
    {
        kernel_name += "_x";
        build_opts.add_option("-DSOFTMAX_X");

        if (is_quantized)
        {
            _tmp_info = TensorInfo(dst_shape, 1, tmp_data_type);
        }
    }
    else
    {
        kernel_name += "_non_x";
        build_opts.add_option("-DSOFTMAX_NON_X");

        TensorShape tmp_shape;

        tmp_shape.set(0, length * vec_size, false);
        tmp_shape.set(1, dst_shape[0] + (vec_size - vec_size_leftover) % vec_size, false);

        for (size_t i = 2; i <= static_cast<size_t>(axis); ++i)
        {
            tmp_shape.set(i, dst_shape[i - 1], false);
        }

        for (size_t i = axis + 1; i < dst_shape.num_dimensions(); ++i)
        {
            tmp_shape.set(i, dst_shape[i], false);
        }

        _tmp_info = TensorInfo(tmp_shape, 1, tmp_data_type);
    }

    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Configure kernel window and kernel arguments.
    Window win = calculate_max_window(src, Steps(vec_size));

    bool has_collapsed = true;

    win = win.shift_dimensions(1, axis); // Remove this axis from the window/GWS.
    win = win.collapse_if_possible(win, 2, has_collapsed);
    ARM_COMPUTE_ERROR_ON(!has_collapsed);

    ICLKernel::configure_internal(win);

    _axis = axis;

    _config_id = "softmax_" + lower_string(string_from_data_type(data_type));
    _config_id += "_" + std::to_string(axis);
    _config_id += "_" + std::to_string(length);
}

void ClSoftmaxKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const auto src =
        utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    auto       dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));
    ICLTensor *tmp = (_tmp_info.total_size() > 0)
                         ? utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_INT_0))
                         : nullptr;

    if (!_prepared)
    {
        _prepared = true;

        const auto *src_info    = src->info();
        const auto *dst_info    = dst->info();
        auto        src_strides = src_info->strides_in_bytes();
        auto        dst_strides = dst_info->strides_in_bytes();

        const auto src_stride_axis = src_strides[_axis];
        const auto dst_stride_axis = dst_strides[_axis];

        // This axis has been removed from execution window, hence we remove it from the list of strides
        // provided to the kernel.
        // In case axis > 0, src/dst_stride_axis will be provided in dedicated argument independent from global ID.
        src_strides.remove(_axis);
        dst_strides.remove(_axis);

        // Argument 0: src_ptr.
        _kernel.setArg<cl_uint>(1, src_strides[0]);
        _kernel.setArg<cl_uint>(2, src_strides[1]);
        _kernel.setArg<cl_uint>(3, src_strides[2]);
        _kernel.setArg<cl_uint>(4, src_info->offset_first_element_in_bytes());

        // Argument 5: dst_ptr.
        _kernel.setArg<cl_uint>(6, dst_strides[0]);
        _kernel.setArg<cl_uint>(7, dst_strides[1]);
        _kernel.setArg<cl_uint>(8, dst_strides[2]);
        _kernel.setArg<cl_uint>(9, dst_info->offset_first_element_in_bytes());

        if (tmp != nullptr)
        {
            const auto *tmp_info    = tmp->info();
            const auto &tmp_strides = tmp_info->strides_in_bytes();

            // Argument 10: tmp_ptr.
            _kernel.setArg<cl_uint>(11, tmp_strides[1]);
            _kernel.setArg<cl_uint>(12, tmp_strides[2]);
            _kernel.setArg<cl_uint>(13, tmp_strides[3]);
            _kernel.setArg<cl_uint>(14, 0);
        }

        if (_axis > 0)
        {
            _kernel.setArg<cl_uint>(15, src_stride_axis);
            _kernel.setArg<cl_uint>(16, dst_stride_axis);
        }
    }

    _kernel.setArg(0, src->cl_buffer());
    _kernel.setArg(5, dst->cl_buffer());

    if (tmp != nullptr)
    {
        _kernel.setArg(10, tmp->cl_buffer());
    }

    enqueue(queue, *this, window, lws_hint());
}

const TensorInfo &ClSoftmaxKernel::tmp_tensor_info() const
{
    return _tmp_info;
}

} // namespace kernels
} // namespace opencl
} // namespace arm_compute
