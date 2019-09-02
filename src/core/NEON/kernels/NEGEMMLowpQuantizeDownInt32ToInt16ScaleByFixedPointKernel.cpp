/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NESymm.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, int min, int max)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::S32);
    ARM_COMPUTE_RETURN_ERROR_ON(max > 32767);
    ARM_COMPUTE_RETURN_ERROR_ON(min < -32768 || min > max);

    // Check biases if exist
    if(bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, bias);
        ARM_COMPUTE_RETURN_ERROR_ON(bias->num_dimensions() > 1);
        ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(0) != bias->dimension(0));
    }

    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::QSYMM16);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(output, input);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output, input->clone()->set_data_type(DataType::QSYMM16));

    // Configure kernel window
    Window win = calculate_max_window(*input, Steps());

    // NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel doesn't need padding so update_window_and_padding() can be skipped
    Coordinates coord;
    coord.set_num_dimensions(output->num_dimensions());
    output->set_valid_region(ValidRegion(coord, output->tensor_shape()));

    return std::make_pair(Status{}, win);
}
} // namespace

class Coordinates;

template <bool is_bounded_relu>
void NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel::run(const Window &window)
{
    const int16x8_t min_s16 = vdupq_n_s16(static_cast<int16_t>(_min));
    const int16x8_t max_s16 = vdupq_n_s16(static_cast<int16_t>(_max));

    ARM_COMPUTE_UNUSED(min_s16);
    ARM_COMPUTE_UNUSED(max_s16);

    const int  window_step_x  = 8;
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator in(_input, win_collapsed);
    Iterator out(_output, win_collapsed);
    if(_bias != nullptr)
    {
        Window win_biases;
        win_biases.set(Window::DimX, Window::Dimension(0, 1, 1));
        win_biases.set(Window::DimY, Window::Dimension(0, 1, 1));

        Iterator bias(_bias, win_biases);
        execute_window_loop(win_collapsed, [&](const Coordinates &)
        {
            // Compute 16 elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                int32x4x2_t in_s32 =
                {
                    {
                        vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 0),
                        vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 4)
                    }
                };

                const int32x4x2_t bias_s32 =
                {
                    {
                        vld1q_s32(reinterpret_cast<const int32_t *>(bias.ptr()) + x + 0),
                        vld1q_s32(reinterpret_cast<const int32_t *>(bias.ptr()) + x + 4)
                    }
                };

                // Add the bias to GEMM's result
                in_s32.val[0] = vaddq_s32(in_s32.val[0], bias_s32.val[0]);
                in_s32.val[1] = vaddq_s32(in_s32.val[1], bias_s32.val[1]);

                vst1q_s16(reinterpret_cast<int16_t *>(out.ptr()) + x, finalize_quantization_int16<is_bounded_relu>(in_s32, _result_fixedpoint_multiplier, _result_shift, min_s16, max_s16));
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                const int32_t bias_value = *(reinterpret_cast<const int32_t *>(bias.ptr()) + x);
                int32_t       in_value   = *(reinterpret_cast<const int32_t *>(in.ptr()) + x);

                // Add bias
                in_value += bias_value;
                // Finalize and store the result
                *(reinterpret_cast<int16_t *>(out.ptr()) + x) = finalize_quantization_int16<is_bounded_relu>(in_value, _result_fixedpoint_multiplier, _result_shift, static_cast<int16_t>(_min),
                                                                                                             static_cast<int16_t>(_max));
            }
        },
        in, out, bias);
    }
    else
    {
        execute_window_loop(win_collapsed, [&](const Coordinates &)
        {
            // Compute 16 elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                int32x4x2_t in_s32 =
                {
                    {
                        vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 0),
                        vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 4)
                    }
                };

                vst1q_s16(reinterpret_cast<int16_t *>(out.ptr()) + x, finalize_quantization_int16<is_bounded_relu>(in_s32, _result_fixedpoint_multiplier, _result_shift, min_s16, max_s16));
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                const int32_t in_value = *(reinterpret_cast<const int32_t *>(in.ptr()) + x);
                ARM_COMPUTE_UNUSED(in_value);
                // Finalize and store the result
                *(reinterpret_cast<int16_t *>(out.ptr()) + x) = finalize_quantization_int16<is_bounded_relu>(in_value, _result_fixedpoint_multiplier, _result_shift, static_cast<int16_t>(_min),
                                                                                                             static_cast<int16_t>(_max));
            }
        },
        in, out);
    }
}

NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel::NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel()
    : _func(nullptr), _input(nullptr), _bias(nullptr), _output(nullptr), _result_fixedpoint_multiplier(0), _result_shift(0), _min(0), _max(0)
{
}

void NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel::configure(const ITensor *input, const ITensor *bias, ITensor *output, int result_fixedpoint_multiplier, int result_shift,
                                                                          int min, int max)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), (bias != nullptr) ? bias->info() : nullptr, output->info(), min, max));

    _input                        = input;
    _bias                         = bias;
    _output                       = output;
    _result_fixedpoint_multiplier = result_fixedpoint_multiplier;
    _result_shift                 = result_shift;
    _min                          = min;
    _max                          = max;

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);

    // Check if we need to clamp the result using min and max
    const bool is_bounded_relu = ((min != max) && !(min == -32768 && max == 32767));
    _func                      = is_bounded_relu ? &NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel::run<true> : &NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel::run<false>;
}

Status NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel::validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, int min, int max)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, bias, output, min, max));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get()).first);

    return Status{};
}

void NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPointKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    (this->*_func)(window);
}
} // namespace arm_compute
