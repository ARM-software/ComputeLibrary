/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "src/core/cpu/kernels/CpuConcatenateDepthKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "src/core/NEON/NEAsymm.h"
#include "src/core/NEON/NEFixedPoint.h"
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
template <typename T>
void depth_concat(const ITensor *src, ITensor *dst, unsigned int depth_offset, const Window &window)
{
    // Offset source
    uint8_t *src_ptr = src->buffer() + src->info()->offset_first_element_in_bytes();

    // Offset destination
    uint8_t *dst_ptr = dst->buffer() + dst->info()->offset_first_element_in_bytes() + depth_offset * dst->info()->strides_in_bytes()[2];

    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());
    const int  window_step_x  = 16 / dst->info()->element_size();

    Window win{ window };
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    win.set(Window::DimZ, Window::Dimension(0, src->info()->tensor_shape().z(), 1));

    Iterator src_it(src, win);
    Iterator dst_it(dst, win);

    const DataType                dt        = src->info()->data_type();
    const UniformQuantizationInfo src_qinfo = src->info()->quantization_info().uniform();
    const UniformQuantizationInfo dst_qinfo = dst->info()->quantization_info().uniform();
    if(dt == DataType::QASYMM8 && src_qinfo != dst_qinfo)
    {
        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto in_ptr  = reinterpret_cast<const uint8_t *>(src_ptr + src_it.offset());
            const auto out_ptr = reinterpret_cast<uint8_t *>(dst_ptr + dst_it.offset());
            int        x       = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                wrapper::vstore(out_ptr + x, vquantize(vdequantize(wrapper::vloadq(in_ptr + x), src_qinfo), dst_qinfo));
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                *(out_ptr + x) = quantize_qasymm8(dequantize_qasymm8(*(in_ptr + x), src_qinfo), dst_qinfo);
            }
        },
        src_it, dst_it);
    }
    else if(dt == DataType::QASYMM8_SIGNED && src_qinfo != dst_qinfo)
    {
        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto in_ptr  = reinterpret_cast<const int8_t *>(src_ptr + src_it.offset());
            const auto out_ptr = reinterpret_cast<int8_t *>(dst_ptr + dst_it.offset());
            int        x       = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                wrapper::vstore(out_ptr + x, vquantize_signed(vdequantize(wrapper::vloadq(in_ptr + x), src_qinfo), dst_qinfo));
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                *(out_ptr + x) = quantize_qasymm8_signed(dequantize_qasymm8_signed(*(in_ptr + x), src_qinfo), dst_qinfo);
            }
        },
        src_it, dst_it);
    }
    else
    {
        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto in_ptr  = reinterpret_cast<const T *>(src_ptr + src_it.offset());
            const auto out_ptr = reinterpret_cast<T *>(dst_ptr + dst_it.offset());
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

Status validate_arguments(const ITensorInfo *input, unsigned int depth_offset, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    //Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input) is not needed here as this kernel doesn't use Neon FP16 instructions.
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(Window::DimX) != output->dimension(Window::DimX));
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(Window::DimY) != output->dimension(Window::DimY));
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(2) + depth_offset > output->dimension(2));
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(3, input, output);

    return Status{};
}
} // namespace

CpuConcatenateDepthKernel::CpuConcatenateDepthKernel()
    : _func(nullptr), _depth_offset(0)
{
}

void CpuConcatenateDepthKernel::configure(const ITensorInfo *src, unsigned int depth_offset, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, depth_offset, dst));

    _func         = nullptr;
    _depth_offset = depth_offset;

    switch(src->data_type())
    {
        case DataType::QASYMM8:
            _func = &depth_concat<uint8_t>;
            break;
        case DataType::QASYMM8_SIGNED:
            _func = &depth_concat<int8_t>;
            break;
        case DataType::F16:
            _func = &depth_concat<uint16_t>;
            break;
        case DataType::F32:
            _func = &depth_concat<uint32_t>;
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported data type.");
    }

    // Configure kernel window
    Window      win = calculate_max_window(*dst, Steps());
    Coordinates coord;
    coord.set_num_dimensions(dst->num_dimensions());

    dst->set_valid_region(ValidRegion(coord, dst->tensor_shape()));
    ICpuKernel::configure(win);
}

Status CpuConcatenateDepthKernel::validate(const arm_compute::ITensorInfo *src,
                                           unsigned int                    depth_offset,
                                           const arm_compute::ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, depth_offset, dst));
    return Status{};
}

void CpuConcatenateDepthKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (*_func)(tensors.get_const_tensor(TensorType::ACL_SRC),
             tensors.get_tensor(TensorType::ACL_DST),
             _depth_offset,
             window);
}

const char *CpuConcatenateDepthKernel::name() const
{
    return "CpuConcatenateDepthKernel";
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
