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
#include "arm_compute/core/NEON/kernels/NEQuantizationLayerKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/NEON/NEAsymm.h"
#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "arm_compute/core/CPP/Validate.h"

#include <arm_neon.h>

using namespace arm_compute;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(output->tensor_shape().total_size() == 0);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::QASYMM8);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);

    return Status{};
}

inline float32x4x4_t load_value(const float *input_ptr)
{
    return { wrapper::vloadq(input_ptr),
             wrapper::vloadq(input_ptr + 4),
             wrapper::vloadq(input_ptr + 8),
             wrapper::vloadq(input_ptr + 12) };
}
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
inline const float32x4x4_t load_value(const float16_t *input_ptr)
{
    return { vcvt_f32_f16(wrapper::vload(input_ptr)),
             vcvt_f32_f16(wrapper::vload(input_ptr + 4)),
             vcvt_f32_f16(wrapper::vload(input_ptr + 8)),
             vcvt_f32_f16(wrapper::vload(input_ptr + 12)) };
}

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
} // namespace

NEQuantizationLayerKernel::NEQuantizationLayerKernel()
    : _input(nullptr), _output(nullptr)
{
}

void NEQuantizationLayerKernel::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info()));

    _input  = input;
    _output = output;

    // Configure kernel window
    Window win_config = calculate_max_window(*input->info(), Steps());

    Coordinates coord;
    coord.set_num_dimensions(output->info()->num_dimensions());
    output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));

    INEKernel::configure(win_config);
}

Status NEQuantizationLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output));

    return Status{};
}

template <typename T>
void NEQuantizationLayerKernel::quantize(const Window &window, const QuantizationInfo &qinfo)
{
    constexpr auto window_step    = 16;
    const auto     window_start_x = static_cast<int>(window.x().start());
    const auto     window_end_x   = static_cast<int>(window.x().end());

#ifdef __aarch64__
    constexpr RoundingPolicy rounding_policy = RoundingPolicy::TO_NEAREST_EVEN;
#else  //__aarch64__
    constexpr RoundingPolicy rounding_policy = RoundingPolicy::TO_ZERO;
#endif //__aarch64__

    // Collapse window and reset first dimension to handle tail calculations manually
    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(_input, win_collapsed);
    Iterator output(_output, win_collapsed);
    execute_window_loop(win_collapsed, [&](const Coordinates &)
    {
        auto input_ptr  = reinterpret_cast<const T *>(input.ptr());
        auto output_ptr = reinterpret_cast<uint8_t *>(output.ptr());

        int x = window_start_x;
        for(; x <= (window_end_x - window_step); x += window_step)
        {
            wrapper::vstore(&output_ptr[x], vquantize(load_value(&input_ptr[x]), qinfo));
        }
        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            output_ptr[x] = qinfo.quantize(input_ptr[x], rounding_policy);
        }
    },
    input, output);
}

void NEQuantizationLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const QuantizationInfo &qinfo = _output->info()->quantization_info();

    switch(_input->info()->data_type())
    {
        case DataType::F32:
            NEQuantizationLayerKernel::quantize<float>(window, qinfo);
            break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            NEQuantizationLayerKernel::quantize<float16_t>(window, qinfo);
            break;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        default:
            ARM_COMPUTE_ERROR("Unsupported data type.");
    }
}
