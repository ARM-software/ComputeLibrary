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
#include "src/core/cpu/kernels/CpuConcatenateHeightKernel.h"

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
Status validate_arguments(const ITensorInfo *src, unsigned int height_offset, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    // Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src) is not needed here as this kernel doesn't use Neon FP16 instructions.
    ARM_COMPUTE_RETURN_ERROR_ON(src->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON(src->dimension(Window::DimX) != dst->dimension(Window::DimX));
    ARM_COMPUTE_RETURN_ERROR_ON(src->dimension(Window::DimY) + height_offset > dst->dimension(Window::DimY));
    for(size_t i = 2; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(src->dimension(i) != dst->dimension(i));
    }

    return Status{};
}
} // namespace

CpuConcatenateHeightKernel::CpuConcatenateHeightKernel()
    : _height_offset(0)
{
}

void CpuConcatenateHeightKernel::configure(const ITensorInfo *src, unsigned int height_offset, ITensorInfo *dst)
{
    ARM_COMPUTE_UNUSED(src);
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, height_offset, dst));

    _height_offset = height_offset;

    // Configure kernel window
    Window      win = calculate_max_window(*dst, Steps());
    Coordinates coord;
    coord.set_num_dimensions(dst->num_dimensions());
    dst->set_valid_region(ValidRegion(coord, dst->tensor_shape()));
    ICpuKernel::configure(win);
}

Status CpuConcatenateHeightKernel::validate(const ITensorInfo *src, unsigned int height_offset, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, height_offset, dst));
    return Status{};
}

void CpuConcatenateHeightKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    const auto src = tensors.get_const_tensor(TensorType::ACL_SRC);
    auto       dst = tensors.get_tensor(TensorType::ACL_DST);

    // Offset destination pointer to the correct position
    uint8_t *dst_ptr = dst->buffer() + dst->info()->offset_first_element_in_bytes() + _height_offset * dst->info()->strides_in_bytes()[Window::DimY];

    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end()) * static_cast<int>(dst->info()->element_size());
    const int  window_step_x  = 16;

    Window win{ window };
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    win.set(Window::DimY, Window::Dimension(0, src->info()->tensor_shape().y(), 1));

    // Create iterators
    Iterator src_it(src, win);
    Iterator dst_it(dst, win);

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
                         vquantize_signed(vdequantize(vld1q_s8(reinterpret_cast<int8_t *>(src_it.ptr()) + x), src_qinfo), dst_qinfo));
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

            int x = window_start_x;
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

const char *CpuConcatenateHeightKernel::name() const
{
    return "CpuConcatenateHeightKernel";
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
