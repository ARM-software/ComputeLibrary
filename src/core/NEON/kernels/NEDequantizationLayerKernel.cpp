/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEDequantizationLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/NEON/NEAsymm.h"
#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8);

    if(output->tensor_shape().total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(output);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F16, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
    }

    return Status{};
}

std::tuple<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    // Configure kernel window
    Window win = calculate_max_window(*input, Steps());

    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*output, input->tensor_shape(), 1, DataType::F32);

    // NEDequantizationLayerKernel doesn't need padding so update_window_and_padding() can be skipped
    Coordinates coord;
    coord.set_num_dimensions(output->num_dimensions());
    output->set_valid_region(ValidRegion(coord, output->tensor_shape()));

    return std::make_tuple(Status{}, win);
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
void run_dequantization(const ITensor *input, ITensor *output, const Window &window)
{
    const QuantizationInfo &qinfo = input->info()->quantization_info();

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
        const auto in_ptr  = reinterpret_cast<const uint8_t *>(in.ptr());
        const auto out_ptr = reinterpret_cast<T *>(out.ptr());

        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const auto vin  = wrapper::vloadq(in_ptr + x);
            const auto vdeq = vdequantize(vin, qinfo);

            store_result<T>(reinterpret_cast<T *>(out_ptr + x), vdeq);
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            uint8_t val    = *(in_ptr + x);
            *(out_ptr + x) = static_cast<T>(qinfo.dequantize(val));
        }
    },
    in, out);
}
} // namespace

NEDequantizationLayerKernel::NEDequantizationLayerKernel()
    : _input(nullptr), _output(nullptr)
{
}

void NEDequantizationLayerKernel::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info()));

    _input  = input;
    _output = output;

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info());

    ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

    INEKernel::configure(std::get<1>(win_config));
}

Status NEDequantizationLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output));
    ARM_COMPUTE_RETURN_ON_ERROR(std::get<0>(validate_and_configure_window(input->clone().get(), output->clone().get())));
    return Status{};
}

void NEDequantizationLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    switch(_output->info()->data_type())
    {
        case DataType::F32:
            run_dequantization<float>(_input, _output, window);
            break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            run_dequantization<float16_t>(_input, _output, window);
            break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        default:
            ARM_COMPUTE_ERROR("Unsupported data type.");
    }
}
} // namespace arm_compute