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
#include "arm_compute/core/NEON/kernels/NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

using namespace arm_compute;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, int min, int max)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::S32);
    ARM_COMPUTE_RETURN_ERROR_ON(max > 255);
    ARM_COMPUTE_RETURN_ERROR_ON(min < 0 || min > max);

    // Check biases if exist
    if(bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, bias);
        ARM_COMPUTE_RETURN_ERROR_ON(bias->num_dimensions() > 1);
        ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(0) != bias->dimension(0));
    }

    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::QASYMM8);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *bias, ITensorInfo *output)
{
    // Note: This kernel performs 16 elements per iteration.
    // However, since we use a left-over for loop, we cannot have any read or write out of memory
    // For this reason num_elems_processed_per_iteration is set to 1
    constexpr unsigned int num_elems_processed_per_iteration = 1;

    // Configure kernel window
    Window win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);

    bool window_changed = update_window_and_padding(win,
                                                    input_access);

    if(output->total_size() != 0)
    {
        AccessWindowHorizontal output_result_access(output, 0, num_elems_processed_per_iteration);
        window_changed = window_changed || update_window_and_padding(win, output_result_access);

        output_result_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));
    }

    if(bias != nullptr)
    {
        AccessWindowStatic bias_access(bias, 0, 0, bias->dimension(0), bias->dimension(1));
        window_changed = window_changed || update_window_and_padding(win, bias_access);
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

inline void scale_input(int32x4x4_t &in_s32, int32x4_t result_offset_s32, int32_t result_mult_int)
{
    // Add the offset terms to GEMM's result
    in_s32.val[0] = vaddq_s32(in_s32.val[0], result_offset_s32);
    in_s32.val[1] = vaddq_s32(in_s32.val[1], result_offset_s32);
    in_s32.val[2] = vaddq_s32(in_s32.val[2], result_offset_s32);
    in_s32.val[3] = vaddq_s32(in_s32.val[3], result_offset_s32);

    // Multiply by result_mult_int
    in_s32.val[0] = vmulq_n_s32(in_s32.val[0], result_mult_int);
    in_s32.val[1] = vmulq_n_s32(in_s32.val[1], result_mult_int);
    in_s32.val[2] = vmulq_n_s32(in_s32.val[2], result_mult_int);
    in_s32.val[3] = vmulq_n_s32(in_s32.val[3], result_mult_int);
}

template <bool    is_bounded_relu>
inline uint8x16_t finalize_quantization(int32x4x4_t &in_s32, int32x4_t result_shift_s32, uint8x16_t min_u8, uint8x16_t max_u8)
{
    const static int32x4_t zero_s32 = vdupq_n_s32(0);

    // Shift final result (negative value shift right)
    in_s32.val[0] = vshlq_s32(in_s32.val[0], result_shift_s32);
    in_s32.val[1] = vshlq_s32(in_s32.val[1], result_shift_s32);
    in_s32.val[2] = vshlq_s32(in_s32.val[2], result_shift_s32);
    in_s32.val[3] = vshlq_s32(in_s32.val[3], result_shift_s32);

    // Saturate negative values
    in_s32.val[0] = vmaxq_s32(in_s32.val[0], zero_s32);
    in_s32.val[1] = vmaxq_s32(in_s32.val[1], zero_s32);
    in_s32.val[2] = vmaxq_s32(in_s32.val[2], zero_s32);
    in_s32.val[3] = vmaxq_s32(in_s32.val[3], zero_s32);

    // Convert S32 to S16
    const int16x8x2_t in_s16 =
    {
        {
            vcombine_s16(vqmovn_s32(in_s32.val[0]), vqmovn_s32(in_s32.val[1])),
            vcombine_s16(vqmovn_s32(in_s32.val[2]), vqmovn_s32(in_s32.val[3]))
        }
    };

    // Convert S16 to U8
    uint8x16_t out_u8 = vcombine_u8(vqmovun_s16(in_s16.val[0]), vqmovun_s16(in_s16.val[1]));

    if(is_bounded_relu)
    {
        out_u8 = vmaxq_u8(out_u8, min_u8);
        out_u8 = vminq_u8(out_u8, max_u8);
    }

    return out_u8;
}
} // namespace

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

template <bool is_bounded_relu>
void NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel::run(const Window &window)
{
    const int32x4_t  result_offset_s32 = vdupq_n_s32(_result_offset);
    const int32x4_t  result_shift_s32  = vdupq_n_s32(-_result_shift);
    const uint8x16_t min_u8            = vdupq_n_u8(static_cast<uint8_t>(_min));
    const uint8x16_t max_u8            = vdupq_n_u8(static_cast<uint8_t>(_max));

    ARM_COMPUTE_UNUSED(min_u8);
    ARM_COMPUTE_UNUSED(max_u8);

    const int  window_step_x  = 16;
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    Window win(window);
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator in(_input, win);
    Iterator out(_output, win);

    if(_bias != nullptr)
    {
        Window win_biases;
        win_biases.set(Window::DimX, Window::Dimension(0, 1, 1));
        win_biases.set(Window::DimY, Window::Dimension(0, 1, 1));

        Iterator bias(_bias, win_biases);
        execute_window_loop(win, [&](const Coordinates &)
        {
            // Compute 16 elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                int32x4x4_t in_s32 =
                {
                    {
                        vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 0),
                        vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 4),
                        vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 8),
                        vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 12)
                    }
                };

                const int32x4x4_t bias_s32 =
                {
                    {
                        vld1q_s32(reinterpret_cast<const int32_t *>(bias.ptr()) + x + 0),
                        vld1q_s32(reinterpret_cast<const int32_t *>(bias.ptr()) + x + 4),
                        vld1q_s32(reinterpret_cast<const int32_t *>(bias.ptr()) + x + 8),
                        vld1q_s32(reinterpret_cast<const int32_t *>(bias.ptr()) + x + 12)
                    }
                };

                // Add the bias to GEMM's result
                in_s32.val[0] = vaddq_s32(in_s32.val[0], bias_s32.val[0]);
                in_s32.val[1] = vaddq_s32(in_s32.val[1], bias_s32.val[1]);
                in_s32.val[2] = vaddq_s32(in_s32.val[2], bias_s32.val[2]);
                in_s32.val[3] = vaddq_s32(in_s32.val[3], bias_s32.val[3]);

                // Add the offset terms to GEMM's result and multiply by result_mult_int
                scale_input(in_s32, result_offset_s32, _result_mult_int);

                vst1q_u8(out.ptr() + x, finalize_quantization<is_bounded_relu>(in_s32, result_shift_s32, min_u8, max_u8));
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                const int bias_value = *(reinterpret_cast<const int *>(bias.ptr()) + x);
                int       in_value   = *(reinterpret_cast<const int *>(in.ptr()) + x);

                // Quantize
                in_value = ((in_value + bias_value + _result_offset) * _result_mult_int) >> _result_shift;

                // Finalize and store the result
                if(is_bounded_relu)
                {
                    *(out.ptr() + x) = static_cast<uint8_t>(std::max(_min, std::min(_max, in_value)));
                }
                else
                {
                    *(out.ptr() + x) = static_cast<uint8_t>(std::max(0, std::min(255, in_value)));
                }
            }
        },
        in, bias, out);
    }
    else
    {
        execute_window_loop(win, [&](const Coordinates &)
        {
            // Compute 16 elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                int32x4x4_t in_s32 =
                {
                    {
                        vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 0),
                        vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 4),
                        vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 8),
                        vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + x + 12)
                    }
                };

                // Add the offset terms to GEMM's result and multiply by result_mult_int
                scale_input(in_s32, result_offset_s32, _result_mult_int);

                vst1q_u8(out.ptr() + x, finalize_quantization<is_bounded_relu>(in_s32, result_shift_s32, min_u8, max_u8));
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                int in_value = *(reinterpret_cast<const int *>(in.ptr()) + x);

                // Quantize
                in_value = ((in_value + _result_offset) * _result_mult_int) >> _result_shift;

                // Finalize and store the result
                if(is_bounded_relu)
                {
                    *(out.ptr() + x) = static_cast<uint8_t>(std::max(_min, std::min(_max, in_value)));
                }
                else
                {
                    *(out.ptr() + x) = static_cast<uint8_t>(std::max(0, std::min(255, in_value)));
                }
            }
        },
        in, out);
    }
}

NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel::NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel()
    : _func(nullptr), _input(nullptr), _bias(nullptr), _output(nullptr), _result_offset(0), _result_mult_int(0), _result_shift(0), _min(0), _max(0)
{
}

void NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel::configure(const ITensor *input, const ITensor *bias, ITensor *output, int result_offset, int result_mult_int, int result_shift, int min, int max)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->clone()->set_data_type(DataType::QASYMM8));

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(),
                                                  (bias != nullptr) ? bias->info() : nullptr,
                                                  output->info(),
                                                  min,
                                                  max));

    _input           = input;
    _bias            = bias;
    _output          = output;
    _result_offset   = result_offset;
    _result_mult_int = result_mult_int;
    _result_shift    = result_shift;
    _min             = min;
    _max             = max;

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), (bias != nullptr) ? bias->info() : nullptr, output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);

    // Check if we need to clamp the result using min and max
    const bool is_bounded_relu = ((min != max) && !(min == 0 && max == 255));
    _func                      = is_bounded_relu ? &NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel::run<true> : &NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel::run<false>;
}

Status NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel::validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, int min, int max)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, bias, output, min, max));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(),
                                                              (bias != nullptr) ? bias->clone().get() : nullptr,
                                                              output->clone().get())
                                .first);

    return Status{};
}

void NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    (this->*_func)(window);
}
