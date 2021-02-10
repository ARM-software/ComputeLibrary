/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "src/core/cpu/kernels/CpuConcatenateWidthKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "src/core/NEON/NEAsymm.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <cstdint>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
Status validate_arguments(const ITensorInfo *src, unsigned int width_offset, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    // Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src) is not needed here as this kernel doesn't use Neon FP16 instructions.
    ARM_COMPUTE_RETURN_ERROR_ON(src->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON(src->dimension(0) + width_offset > dst->dimension(0));

    for(size_t i = 1; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(src->dimension(i) != dst->dimension(i));
    }

    return Status{};
}
} // namespace

CpuConcatenateWidthKernel::CpuConcatenateWidthKernel()
    : _width_offset(0)
{
}

void CpuConcatenateWidthKernel::configure(const ITensorInfo *src, unsigned int width_offset, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, width_offset, dst));

    _width_offset = width_offset;

    // Configure kernel window
    Window      win = calculate_max_window(*src, Steps());
    Coordinates coord;
    coord.set_num_dimensions(dst->num_dimensions());
    dst->set_valid_region(ValidRegion(coord, dst->tensor_shape()));

    ICpuKernel::configure(win);
}

Status CpuConcatenateWidthKernel::validate(const ITensorInfo *src, unsigned int width_offset, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, width_offset, dst));
    return Status{};
}

void CpuConcatenateWidthKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    const auto src = tensors.get_const_tensor(TensorType::ACL_SRC);
    auto       dst = tensors.get_tensor(TensorType::ACL_DST);

    // Offset output pointer to the correct position
    uint8_t *dst_ptr = dst->buffer() + dst->info()->offset_first_element_in_bytes() + _width_offset * dst->info()->strides_in_bytes()[0];

    const auto    window_start_x = static_cast<int>(window.x().start());
    const auto    window_end_x   = static_cast<int>(window.x().end()) * static_cast<int>(dst->info()->element_size());
    constexpr int window_step_x  = 16;

    Window win{ window };
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    // Create iterators
    Iterator                       src_it(src, win);
    Iterator                       dst_it(dst, win);
    const DataType                 dt        = src->info()->data_type();
    const UniformQuantizationInfo &src_qinfo = src->info()->quantization_info().uniform();
    const UniformQuantizationInfo &dst_qinfo = dst->info()->quantization_info().uniform();
    if(dt == DataType::QASYMM8 && src_qinfo != dst_qinfo)
    {
        execute_window_loop(win, [&](const Coordinates &)
        {
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                vst1q_u8(dst_ptr + dst_it.offset() + x, vquantize(vdequantize(vld1q_u8(src_it.ptr() + x), src_qinfo), dst_qinfo));
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                *(dst_ptr + dst_it.offset() + x) = quantize_qasymm8(dequantize_qasymm8(*(src_it.ptr() + x), src_qinfo), dst_qinfo);
            }
        },
        src_it, dst_it);
    }
    else if(dt == DataType::QASYMM8_SIGNED && src_qinfo != dst_qinfo)
    {
        execute_window_loop(win, [&](const Coordinates &)
        {
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                vst1q_s8(reinterpret_cast<int8_t *>(dst_ptr + dst_it.offset() + x),
                         vquantize_signed(vdequantize(vld1q_s8(reinterpret_cast<int8_t *>(src_it.ptr() + x)), src_qinfo), dst_qinfo));
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                *(dst_ptr + dst_it.offset() + x) = quantize_qasymm8_signed(dequantize_qasymm8_signed(*(src_it.ptr() + x), src_qinfo), dst_qinfo);
            }
        },
        src_it, dst_it);
    }
    else
    {
        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto in_ptr  = src_it.ptr();
            const auto out_ptr = dst_ptr + dst_it.offset();
            int        x       = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                wrapper::vstore(out_ptr + x, wrapper::vloadq(in_ptr + x));
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                *(out_ptr + x) = *(in_ptr + x);
            }
        },
        src_it, dst_it);
    }
}

const char *CpuConcatenateWidthKernel::name() const
{
    return "CpuConcatenateWidthKernel";
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
