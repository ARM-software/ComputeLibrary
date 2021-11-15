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
#include "src/cpu/kernels/CpuDequantizeKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "src/core/CPP/Validate.h"
#include "src/core/NEON/NEAsymm.h"
#include "src/core/NEON/NESymm.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
Status validate_arguments(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::QSYMM8_PER_CHANNEL, DataType::QSYMM8, DataType::QSYMM16);

    if(dst->tensor_shape().total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(dst);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, DataType::F16, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, dst);
    }

    return Status{};
}

template <typename T>
inline void store_result(T *ptr, const float32x4x4_t &v)
{
    ARM_COMPUTE_UNUSED(ptr, v);
}

template <>
inline void store_result<float>(float *ptr, const float32x4x4_t &v)
{
    wrapper::vstore(ptr, v.val[0]);
    wrapper::vstore(ptr + 4, v.val[1]);
    wrapper::vstore(ptr + 8, v.val[2]);
    wrapper::vstore(ptr + 12, v.val[3]);
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <>
inline void store_result<float16_t>(float16_t *ptr, const float32x4x4_t &v)
{
    wrapper::vstore(ptr, vcombine_f16(vcvt_f16_f32(v.val[0]), vcvt_f16_f32(v.val[1])));
    wrapper::vstore(ptr + 8, vcombine_f16(vcvt_f16_f32(v.val[2]), vcvt_f16_f32(v.val[3])));
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

template <typename T>
inline void store_result(T *ptr, const float32x4x2_t &v)
{
    ARM_COMPUTE_UNUSED(ptr, v);
}

template <>
inline void store_result<float>(float *ptr, const float32x4x2_t &v)
{
    wrapper::vstore(ptr, v.val[0]);
    wrapper::vstore(ptr + 4, v.val[1]);
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <>
inline void store_result<float16_t>(float16_t *ptr, const float32x4x2_t &v)
{
    wrapper::vstore(ptr, vcombine_f16(vcvt_f16_f32(v.val[0]), vcvt_f16_f32(v.val[1])));
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

template <typename TOut, typename TIn>
void run_dequantization_qasymm8(const ITensor *input, ITensor *output, const Window &window)
{
    const UniformQuantizationInfo &qinfo  = input->info()->quantization_info().uniform();
    const float                    scale  = qinfo.scale;
    const int32_t                  offset = qinfo.offset;

    const int  window_step_x  = 16;
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    // Collapse window and reset first dimension to handle tail calculations manually
    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    // Create iterators
    Iterator in(input, win_collapsed);
    Iterator out(output, win_collapsed);

    execute_window_loop(win_collapsed, [&](const Coordinates &)
    {
        const auto in_ptr  = reinterpret_cast<const TIn *>(in.ptr());
        const auto out_ptr = reinterpret_cast<TOut *>(out.ptr());

        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const auto vin  = wrapper::vloadq(in_ptr + x);
            const auto vdeq = vdequantize(vin, scale, offset);

            store_result(reinterpret_cast<TOut *>(out_ptr + x), vdeq);
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            auto val       = *(in_ptr + x);
            *(out_ptr + x) = static_cast<TOut>(Qasymm8QuantizationHelper<TIn>::dequantize(val, qinfo));
        }
    },
    in, out);
}

template <typename T>
void run_dequantization_qsymm8_per_channel_nchw(const ITensor *input, ITensor *output, const Window &window)
{
    const auto scale = input->info()->quantization_info().scale();

    const int  window_step_x  = 16;
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    // Reset first dimension to handle tail calculations manually
    Window win(window);
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    // Create iterators
    Iterator in(input, win);
    Iterator out(output, win);

    execute_window_loop(win, [&](const Coordinates & id)
    {
        const auto in_ptr  = reinterpret_cast<const int8_t *>(in.ptr());
        const auto out_ptr = reinterpret_cast<T *>(out.ptr());

        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const auto vin  = wrapper::vloadq(in_ptr + x);
            const auto vdeq = vdequantize(vin, scale[id.z()]);

            store_result<T>(reinterpret_cast<T *>(out_ptr + x), vdeq);
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            int8_t val     = *(in_ptr + x);
            *(out_ptr + x) = static_cast<T>(dequantize(val, scale[id.z()]));
        }
    },
    in, out);
}

template <typename T>
void run_dequantization_qsymm8_per_channel_nhwc(const ITensor *input, ITensor *output, const Window &window)
{
    const auto scale = input->info()->quantization_info().scale();

    const int  window_step_x  = 16;
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    // Reset first dimension to handle tail calculations manually
    Window win(window);
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    // Create iterators
    Iterator in(input, win);
    Iterator out(output, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto in_ptr  = reinterpret_cast<const int8_t *>(in.ptr());
        const auto out_ptr = reinterpret_cast<T *>(out.ptr());

        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const float32x4x4_t vscale =
            {
                {
                    scale[x + 0], scale[x + 1], scale[x + 2], scale[x + 3],
                    scale[x + 4], scale[x + 5], scale[x + 6], scale[x + 7],
                    scale[x + 8], scale[x + 9], scale[x + 10], scale[x + 11],
                    scale[x + 12], scale[x + 13], scale[x + 14], scale[x + 15]
                }
            };
            const auto vin  = wrapper::vloadq(in_ptr + x);
            const auto vdeq = vdequantize(vin, vscale);

            store_result<T>(reinterpret_cast<T *>(out_ptr + x), vdeq);
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            int8_t val     = *(in_ptr + x);
            *(out_ptr + x) = static_cast<T>(dequantize(val, scale[x]));
        }
    },
    in, out);
}

template <typename T>
void run_dequantization_qsymm8(const ITensor *input, ITensor *output, const Window &window)
{
    const UniformQuantizationInfo &qinfo = input->info()->quantization_info().uniform();
    const float                    scale = qinfo.scale;

    const int  window_step_x  = 16;
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    // Collapse window and reset first dimension to handle tail calculations manually
    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    // Create iterators
    Iterator in(input, win_collapsed);
    Iterator out(output, win_collapsed);

    execute_window_loop(win_collapsed, [&](const Coordinates &)
    {
        const auto in_ptr  = reinterpret_cast<const int8_t *>(in.ptr());
        const auto out_ptr = reinterpret_cast<T *>(out.ptr());

        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const auto vin  = wrapper::vloadq(in_ptr + x);
            const auto vdeq = vdequantize(vin, scale);

            store_result<T>(reinterpret_cast<T *>(out_ptr + x), vdeq);
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            int8_t val     = *(in_ptr + x);
            *(out_ptr + x) = static_cast<T>(dequantize(val, scale));
        }
    },
    in, out);
}

template <typename T>
void run_dequantization_qsymm16(const ITensor *input, ITensor *output, const Window &window)
{
    const UniformQuantizationInfo &qinfo = input->info()->quantization_info().uniform();
    const float                    scale = qinfo.scale;

    const int  window_step_x  = 8;
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    // Collapse window and reset first dimension to handle tail calculations manually
    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    // Create iterators
    Iterator in(input, win_collapsed);
    Iterator out(output, win_collapsed);

    execute_window_loop(win_collapsed, [&](const Coordinates &)
    {
        const auto in_ptr  = reinterpret_cast<const int16_t *>(in.ptr());
        const auto out_ptr = reinterpret_cast<T *>(out.ptr());

        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const auto vin  = wrapper::vloadq(in_ptr + x);
            const auto vdeq = vdequantize_int16(vin, scale);

            store_result<T>(reinterpret_cast<T *>(out_ptr + x), vdeq);
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            int16_t val    = *(in_ptr + x);
            *(out_ptr + x) = static_cast<T>(dequantize_qsymm16(val, scale));
        }
    },
    in, out);
}

template <typename T>
void run_dequantization_core(const ITensor *input, ITensor *output, const Window &window)
{
    switch(input->info()->data_type())
    {
        case DataType::QASYMM8:
            run_dequantization_qasymm8<T, uint8_t>(input, output, window);
            break;
        case DataType::QASYMM8_SIGNED:
            run_dequantization_qasymm8<T, int8_t>(input, output, window);
            break;
        case DataType::QSYMM8_PER_CHANNEL:
            input->info()->data_layout() == DataLayout::NHWC ? run_dequantization_qsymm8_per_channel_nhwc<T>(input, output, window) : run_dequantization_qsymm8_per_channel_nchw<T>(input, output, window);
            break;
        case DataType::QSYMM8:
            run_dequantization_qsymm8<T>(input, output, window);
            break;
        case DataType::QSYMM16:
            run_dequantization_qsymm16<T>(input, output, window);
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported data type.");
    }
}
} // namespace

void CpuDequantizeKernel::configure(const ITensorInfo *src, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst));

    // Configure kernel window
    Window win = calculate_max_window(*src, Steps());

    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*dst, src->tensor_shape(), 1, DataType::F32);

    ICpuKernel::configure(win);
}

Status CpuDequantizeKernel::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst));
    return Status{};
}

void CpuDequantizeKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    const auto src = tensors.get_const_tensor(TensorType::ACL_SRC);
    auto       dst = tensors.get_tensor(TensorType::ACL_DST);

    switch(dst->info()->data_type())
    {
        case DataType::F32:
            run_dequantization_core<float>(src, dst, window);
            break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            run_dequantization_core<float16_t>(src, dst, window);
            break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        default:
            ARM_COMPUTE_ERROR("Unsupported data type.");
    }
}
const char *CpuDequantizeKernel::name() const
{
    return "CpuDequantizeKernel";
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
