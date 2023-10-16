/*
 * Copyright (c) 2017-2023 Arm Limited.
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
#include "src/cpu/kernels/CpuIm2ColKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/Validate.h"

#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/directconv2d/impl.h"
#include "src/cpu/kernels/directconv2d/list.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <tuple>

namespace arm_compute
{
using namespace misc::shape_calculator;
namespace cpu
{
namespace kernels
{
void run_im2col_fp32_pad(const ITensor                        *src,
                         ITensor                              *dst,
                         const Window                         &window,
                         DataLayout                            data_layout,
                         const PadStrideInfo                  &conv_info,
                         std::pair<unsigned int, unsigned int> convolved_dims,
                         const Size2D                         &kernel_dims,
                         const Size2D                         &dilation,
                         uint32_t                              input_pad_right,
                         bool                                  has_bias)
{
    arm_compute::cpu::kernels::run_im2col<float, true, false>(src, dst, window, data_layout, conv_info, convolved_dims,
                                                              kernel_dims, dilation, input_pad_right, has_bias);
}

void run_im2col_fp32_nopad(const ITensor                        *src,
                           ITensor                              *dst,
                           const Window                         &window,
                           DataLayout                            data_layout,
                           const PadStrideInfo                  &conv_info,
                           std::pair<unsigned int, unsigned int> convolved_dims,
                           const Size2D                         &kernel_dims,
                           const Size2D                         &dilation,
                           uint32_t                              input_pad_right,
                           bool                                  has_bias)
{
    arm_compute::cpu::kernels::run_im2col<float, false, false>(src, dst, window, data_layout, conv_info, convolved_dims,
                                                               kernel_dims, dilation, input_pad_right, has_bias);
}

#if defined(ARM_COMPUTE_ENABLE_BF16)
void run_im2col_bf16_pad(const ITensor                        *src,
                         ITensor                              *dst,
                         const Window                         &window,
                         DataLayout                            data_layout,
                         const PadStrideInfo                  &conv_info,
                         std::pair<unsigned int, unsigned int> convolved_dims,
                         const Size2D                         &kernel_dims,
                         const Size2D                         &dilation,
                         uint32_t                              input_pad_right,
                         bool                                  has_bias)
{
    arm_compute::cpu::kernels::run_im2col<bfloat16, true, false>(
        src, dst, window, data_layout, conv_info, convolved_dims, kernel_dims, dilation, input_pad_right, has_bias);
}

void run_im2col_bf16_nopad(const ITensor                        *src,
                           ITensor                              *dst,
                           const Window                         &window,
                           DataLayout                            data_layout,
                           const PadStrideInfo                  &conv_info,
                           std::pair<unsigned int, unsigned int> convolved_dims,
                           const Size2D                         &kernel_dims,
                           const Size2D                         &dilation,
                           uint32_t                              input_pad_right,
                           bool                                  has_bias)
{
    arm_compute::cpu::kernels::run_im2col<bfloat16, false, false>(
        src, dst, window, data_layout, conv_info, convolved_dims, kernel_dims, dilation, input_pad_right, has_bias);
}
#endif /* defined(ARM_COMPUTE_ENABLE_BF16) */

void run_im2col_int8_nopad_nhwc(const ITensor                        *src,
                                ITensor                              *dst,
                                const Window                         &window,
                                DataLayout                            data_layout,
                                const PadStrideInfo                  &conv_info,
                                std::pair<unsigned int, unsigned int> convolved_dims,
                                const Size2D                         &kernel_dims,
                                const Size2D                         &dilation,
                                uint32_t                              input_pad_right,
                                bool                                  has_bias)
{
    arm_compute::cpu::kernels::run_im2col<int8_t, false, false>(
        src, dst, window, data_layout, conv_info, convolved_dims, kernel_dims, dilation, input_pad_right, has_bias);
}

void run_im2col_uint8_nopad_nhwc(const ITensor                        *src,
                                 ITensor                              *dst,
                                 const Window                         &window,
                                 DataLayout                            data_layout,
                                 const PadStrideInfo                  &conv_info,
                                 std::pair<unsigned int, unsigned int> convolved_dims,
                                 const Size2D                         &kernel_dims,
                                 const Size2D                         &dilation,
                                 uint32_t                              input_pad_right,
                                 bool                                  has_bias)
{
    arm_compute::cpu::kernels::run_im2col<uint8_t, false, false>(
        src, dst, window, data_layout, conv_info, convolved_dims, kernel_dims, dilation, input_pad_right, has_bias);
}

void run_im2col_qasymm8_pad_nhwc(const ITensor                        *src,
                                 ITensor                              *dst,
                                 const Window                         &window,
                                 DataLayout                            data_layout,
                                 const PadStrideInfo                  &conv_info,
                                 std::pair<unsigned int, unsigned int> convolved_dims,
                                 const Size2D                         &kernel_dims,
                                 const Size2D                         &dilation,
                                 uint32_t                              input_pad_right,
                                 bool                                  has_bias)
{
    arm_compute::cpu::kernels::run_im2col<qasymm8_t, true, false>(
        src, dst, window, data_layout, conv_info, convolved_dims, kernel_dims, dilation, input_pad_right, has_bias);
}

void internal_run_im2col_fp16_pad(const ITensor                        *src,
                                  ITensor                              *dst,
                                  const Window                         &window,
                                  DataLayout                            data_layout,
                                  const PadStrideInfo                  &conv_info,
                                  std::pair<unsigned int, unsigned int> convolved_dims,
                                  const Size2D                         &kernel_dims,
                                  const Size2D                         &dilation,
                                  uint32_t                              input_pad_right,
                                  bool                                  has_bias)
{
/*
   Note that when building with the option data_type_support=fp32 the fp16.cpp files won't be compiled and the linker
   would fail with the error undefined arm_compute::cpu::kernels::run_im2col_fp16_pad.
   To avoid this problem we only call to the actual fp16 kernel if ENABLE_FP16_KERNELS is defined.
*/
#if defined(ENABLE_FP16_KERNELS)
    arm_compute::cpu::kernels::run_im2col_fp16_pad(src, dst, window, data_layout, conv_info, convolved_dims,
                                                   kernel_dims, dilation, input_pad_right, has_bias);
#else  // defined(ENABLE_FP16_KERNELS)
    ARM_COMPUTE_UNUSED(src, dst, window, data_layout, conv_info, convolved_dims, kernel_dims, dilation, input_pad_right,
                       has_bias);
#endif // defined(ENABLE_FP16_KERNELS)
}

void internal_run_im2col_fp16_nopad(const ITensor                        *src,
                                    ITensor                              *dst,
                                    const Window                         &window,
                                    DataLayout                            data_layout,
                                    const PadStrideInfo                  &conv_info,
                                    std::pair<unsigned int, unsigned int> convolved_dims,
                                    const Size2D                         &kernel_dims,
                                    const Size2D                         &dilation,
                                    uint32_t                              input_pad_right,
                                    bool                                  has_bias)
{
#if defined(ENABLE_FP16_KERNELS)
    arm_compute::cpu::kernels::run_im2col_fp16_nopad(src, dst, window, data_layout, conv_info, convolved_dims,
                                                     kernel_dims, dilation, input_pad_right, has_bias);
#else  // defined(ENABLE_FP16_KERNELS)
    ARM_COMPUTE_UNUSED(src, dst, window, data_layout, conv_info, convolved_dims, kernel_dims, dilation, input_pad_right,
                       has_bias);
#endif // defined(ENABLE_FP16_KERNELS)
}

void internal_run_im2col_fp16_nchw_pad(const ITensor                        *src,
                                       ITensor                              *dst,
                                       const Window                         &window,
                                       DataLayout                            data_layout,
                                       const PadStrideInfo                  &conv_info,
                                       std::pair<unsigned int, unsigned int> convolved_dims,
                                       const Size2D                         &kernel_dims,
                                       const Size2D                         &dilation,
                                       uint32_t                              input_pad_right,
                                       bool                                  has_bias)
{
#if defined(ENABLE_FP16_KERNELS)
    arm_compute::cpu::kernels::run_im2col_fp16_nchw_pad(src, dst, window, data_layout, conv_info, convolved_dims,
                                                        kernel_dims, dilation, input_pad_right, has_bias);
#else  // defined(ENABLE_FP16_KERNELS)
    ARM_COMPUTE_UNUSED(src, dst, window, data_layout, conv_info, convolved_dims, kernel_dims, dilation, input_pad_right,
                       has_bias);
#endif // defined(ENABLE_FP16_KERNELS)
}

void internal_run_im2col_fp16_nchw_nopad(const ITensor                        *src,
                                         ITensor                              *dst,
                                         const Window                         &window,
                                         DataLayout                            data_layout,
                                         const PadStrideInfo                  &conv_info,
                                         std::pair<unsigned int, unsigned int> convolved_dims,
                                         const Size2D                         &kernel_dims,
                                         const Size2D                         &dilation,
                                         uint32_t                              input_pad_right,
                                         bool                                  has_bias)
{
#if defined(ENABLE_FP16_KERNELS)
    arm_compute::cpu::kernels::run_im2col_fp16_nchw_nopad(src, dst, window, data_layout, conv_info, convolved_dims,
                                                          kernel_dims, dilation, input_pad_right, has_bias);
#else  //  defined(ENABLE_FP16_KERNELS)
    ARM_COMPUTE_UNUSED(src, dst, window, data_layout, conv_info, convolved_dims, kernel_dims, dilation, input_pad_right,
                       has_bias);
#endif // defined(ENABLE_FP16_KERNELS)
}

namespace
{
Status validate_arguments(const ITensorInfo   *input,
                          const ITensorInfo   *output,
                          const Size2D        &kernel_dims,
                          const PadStrideInfo &conv_info,
                          bool                 has_bias,
                          const Size2D        &dilation,
                          unsigned int         num_groups,
                          unsigned int         input_pad_right)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED,
                                                         DataType::BFLOAT16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(is_data_type_quantized(input->data_type()) && has_bias);
    ARM_COMPUTE_RETURN_ERROR_ON((dilation.x() < 1) || (dilation.y() < 1));
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(num_groups > 1, "Number of groups greater than one are not supported on Neon");

    // Since there's no implicit padding added, check the total input spatial dimensions (with conv paddings) are big enough for the kernel dimensions
    const unsigned int width_idx   = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::WIDTH);
    const unsigned int height_idx  = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::HEIGHT);
    const unsigned     total_width = input->dimension(width_idx) + conv_info.pad_left() + conv_info.pad_right();
    const unsigned     total_height = input->dimension(height_idx) + conv_info.pad_top() + conv_info.pad_bottom();
    ARM_COMPUTE_RETURN_ERROR_ON((total_width < kernel_dims.width) || (total_height < kernel_dims.height));

    if (output->total_size() > 0)
    {
        TensorInfo expected_output = output->clone()->set_tensor_shape(compute_im2col_conv_shape(
            input, kernel_dims, conv_info, has_bias, dilation, false, num_groups, input_pad_right));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&expected_output, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    }

    return Status{};
}
} // namespace

void CpuIm2ColKernel::configure(const ITensorInfo   *src,
                                ITensorInfo         *dst,
                                const Size2D        &kernel_dims,
                                const PadStrideInfo &conv_info,
                                bool                 has_bias,
                                const Size2D        &dilation,
                                unsigned int         num_groups,
                                unsigned int         input_pad_right)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(
        validate_arguments(src, dst, kernel_dims, conv_info, has_bias, dilation, num_groups, input_pad_right));
    ARM_COMPUTE_UNUSED(num_groups);

    _data_layout                   = src->data_layout();
    const unsigned int width_idx   = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::WIDTH);
    const unsigned int height_idx  = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);
    const unsigned int channel_idx = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::CHANNEL);

    _conv_info       = conv_info;
    _kernel_width    = kernel_dims.width;
    _kernel_height   = kernel_dims.height;
    _input_pad_right = input_pad_right;
    _dilation        = dilation;
    _convolved_dims  = scaled_dimensions(src->dimension(width_idx), dst->dimension(height_idx), _kernel_width,
                                         _kernel_height, _conv_info, _dilation);
    _has_bias        = has_bias;

    if (_data_layout == DataLayout::NCHW)
    {
        switch (src->data_type())
        {
            case DataType::F32:
                _func = (!conv_info.has_padding()) ? &run_im2col_fp32_nchw_nopad : &run_im2col_fp32_nchw_pad;
                break;
            case DataType::F16:
                _func = (!conv_info.has_padding()) ? &internal_run_im2col_fp16_nchw_nopad
                                                   : &internal_run_im2col_fp16_nchw_pad;
                break;
#if defined(ARM_COMPUTE_ENABLE_BF16)
            case DataType::BFLOAT16:
                _func = (!conv_info.has_padding()) ? &run_im2col_bf16_nchw_nopad : &run_im2col_bf16_nchw_pad;
                break;
#endif /* defined(ARM_COMPUTE_ENABLE_BF16) */
            case DataType::QASYMM8_SIGNED:
            case DataType::QASYMM8:
                _func = (!conv_info.has_padding()) ? &run_im2col_qasymm8_nchw_nopad : &run_im2col_qasymm8_nchw_pad;
                break;
            default:
                ARM_COMPUTE_ERROR("Data type not supported");
                break;
        }
    }
    else
    {
        switch (src->data_type())
        {
            case DataType::F32:
                _func = (!conv_info.has_padding()) ? &run_im2col_fp32_nopad : &run_im2col_fp32_pad;
                break;
            case DataType::F16:
                _func = (!conv_info.has_padding()) ? &internal_run_im2col_fp16_nopad : &internal_run_im2col_fp16_pad;
                break;
#if defined(ARM_COMPUTE_ENABLE_BF16)
            case DataType::BFLOAT16:
                _func = (!conv_info.has_padding()) ? &run_im2col_bf16_nopad : &run_im2col_bf16_pad;
                break;
#endif /* defined(ARM_COMPUTE_ENABLE_BF16) */
            case DataType::QASYMM8:
                _func = (!conv_info.has_padding()) ? &run_im2col_uint8_nopad_nhwc : &run_im2col_qasymm8_pad_nhwc;
                break;
            case DataType::QASYMM8_SIGNED:
                _func = (!conv_info.has_padding()) ? &run_im2col_int8_nopad_nhwc : &run_im2col_qasymm8_pad_nhwc;
                break;
            default:
                ARM_COMPUTE_ERROR("Data type not supported");
                break;
        }
    }

    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(
        *dst, src->clone()->set_tensor_shape(compute_im2col_conv_shape(src, kernel_dims, conv_info, has_bias, dilation,
                                                                       false, num_groups, _input_pad_right)));

    std::pair<unsigned int, unsigned int> convolved_dims =
        scaled_dimensions(src->dimension(width_idx), src->dimension(height_idx), kernel_dims.width, kernel_dims.height,
                          conv_info, dilation);

    Window win = calculate_max_window(*src, Steps());
    win.set(width_idx, Window::Dimension(0, convolved_dims.first, 1));
    win.set(height_idx, Window::Dimension(0, convolved_dims.second, 1));
    win.set(channel_idx, Window::Dimension(0, 1, 1));
    // Configure kernel window
    ICpuKernel::configure(win);
}

Status CpuIm2ColKernel::validate(const ITensorInfo   *src,
                                 const ITensorInfo   *dst,
                                 const Size2D        &kernel_dims,
                                 const PadStrideInfo &conv_info,
                                 bool                 has_bias,
                                 const Size2D        &dilation,
                                 unsigned int         num_groups,
                                 unsigned int         input_pad_right)
{
    ARM_COMPUTE_RETURN_ON_ERROR(
        validate_arguments(src, dst, kernel_dims, conv_info, has_bias, dilation, num_groups, input_pad_right));
    return Status{};
}

void CpuIm2ColKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    auto src = tensors.get_const_tensor(TensorType::ACL_SRC);
    auto dst = tensors.get_tensor(TensorType::ACL_DST);

    _func(src, dst, window, _data_layout, _conv_info, _convolved_dims, Size2D(_kernel_width, _kernel_height), _dilation,
          _input_pad_right, _has_bias);
}

const char *CpuIm2ColKernel::name() const
{
    return "CpuIm2ColKernel";
}

size_t CpuIm2ColKernel::get_mws(const CPUInfo &platform, size_t thread_count) const
{
    ARM_COMPUTE_UNUSED(thread_count);
    ARM_COMPUTE_UNUSED(platform);

    return ICPPKernel::default_mws;
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
