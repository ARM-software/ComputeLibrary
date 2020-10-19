/*
 * Copyright (c) 2020 Arm Limited.
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
#include "arm_compute/core/NEON/kernels/NEGEMMLowpQuantizeDownInt32ScaleKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

namespace arm_compute
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, const GEMMLowpOutputStageInfo *output_stage)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::S32);

    ARM_COMPUTE_RETURN_ERROR_ON(output_stage->gemmlowp_max_bound > std::get<1>(quantization::get_min_max_values_from_quantized_data_type(output_stage->output_data_type)));
    ARM_COMPUTE_RETURN_ERROR_ON(output_stage->gemmlowp_min_bound < std::get<0>(quantization::get_min_max_values_from_quantized_data_type(output_stage->output_data_type))
                                || output_stage->gemmlowp_min_bound > output_stage->gemmlowp_max_bound);

    // Check biases if exist
    if(bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, bias);
        ARM_COMPUTE_RETURN_ERROR_ON(bias->num_dimensions() > 1);
        ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(0) != bias->dimension(0));
    }

    if(output->total_size() != 0)
    {
        if(output->data_type() != output_stage->output_data_type && (output_stage->output_data_type == DataType::QASYMM8 || output_stage->output_data_type == DataType::QASYMM8_SIGNED))
        {
            ARM_COMPUTE_RETURN_ERROR_MSG("Mismatching data types");
        }

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
    }

    return Status{};
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

template <typename T>
inline typename std::enable_if<std::is_same<T, uint8_t>::value,
       typename wrapper::traits::neon_vector<T, 16>::type>::type
       convert_to_8bit(const int16x8x2_t in_s16)
{
    return wrapper::vcombine(wrapper::vqmovun(in_s16.val[0]), wrapper::vqmovun(in_s16.val[1]));
}

template <typename T>
inline typename std::enable_if<std::is_same<T, int8_t>::value,
       typename wrapper::traits::neon_vector<T, 16>::type>::type
       convert_to_8bit(const int16x8x2_t in_s16)
{
    return wrapper::vcombine(wrapper::vqmovn(in_s16.val[0]), wrapper::vqmovn(in_s16.val[1]));
}

template <typename T>
inline typename wrapper::traits::neon_vector<T, 16>::type finalize_quantization(int32x4x4_t &in_s32, int32x4_t result_shift_s32, typename wrapper::traits::neon_vector<T, 16>::type min,
                                                                                typename wrapper::traits::neon_vector<T, 16>::type max)
{
    // Shift final result (negative value shift right)
    in_s32.val[0] = vshlq_s32(in_s32.val[0], result_shift_s32);
    in_s32.val[1] = vshlq_s32(in_s32.val[1], result_shift_s32);
    in_s32.val[2] = vshlq_s32(in_s32.val[2], result_shift_s32);
    in_s32.val[3] = vshlq_s32(in_s32.val[3], result_shift_s32);

    // Convert S32 to S16
    const int16x8x2_t in_s16 =
    {
        {
            vcombine_s16(vqmovn_s32(in_s32.val[0]), vqmovn_s32(in_s32.val[1])),
            vcombine_s16(vqmovn_s32(in_s32.val[2]), vqmovn_s32(in_s32.val[3]))
        }
    };

    // Convert S16 to S8 or U8
    typename wrapper::traits::neon_vector<T, 16>::type out = convert_to_8bit<T>(in_s16);

    out = wrapper::vmax(out, min);
    out = wrapper::vmin(out, max);

    return out;
}

class Coordinates;

template <typename T>
void NEGEMMLowpQuantizeDownInt32ScaleKernel::run(const Window &window)
{
    using VectorType = typename wrapper::traits::neon_vector<T, 16>::type;

    const int32x4_t result_offset_s32 = vdupq_n_s32(_output_stage->gemmlowp_offset);
    const int32x4_t result_shift_s32  = vdupq_n_s32(-_output_stage->gemmlowp_shift);
    const int       window_step_x     = 16;
    const auto      window_start_x    = static_cast<int>(window.x().start());
    const auto      window_end_x      = static_cast<int>(window.x().end());

    const int clamp_min = (_is_bounded_relu) ? _output_stage->gemmlowp_min_bound : std::numeric_limits<T>::lowest();
    const int clamp_max = (_is_bounded_relu) ? _output_stage->gemmlowp_max_bound : std::numeric_limits<T>::max();

    VectorType min = wrapper::vdup_n(static_cast<T>(clamp_min), wrapper::traits::vector_128_tag{});
    VectorType max = wrapper::vdup_n(static_cast<T>(clamp_max), wrapper::traits::vector_128_tag{});

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
                scale_input(in_s32, result_offset_s32, _output_stage->gemmlowp_multiplier);

                wrapper::vstore(reinterpret_cast<T *>(out.ptr() + x), finalize_quantization<T>(in_s32, result_shift_s32, min, max));
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                const int bias_value = *(reinterpret_cast<const int *>(bias.ptr()) + x);
                int       in_value   = *(reinterpret_cast<const int *>(in.ptr()) + x);

                // Quantize
                in_value = ((in_value + bias_value + _output_stage->gemmlowp_offset) * _output_stage->gemmlowp_multiplier) >> _output_stage->gemmlowp_shift;

                // Store the result
                *(out.ptr() + x) = static_cast<T>(utility::clamp<int>(in_value, clamp_min, clamp_max));
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
                scale_input(in_s32, result_offset_s32, _output_stage->gemmlowp_multiplier);

                wrapper::vstore(reinterpret_cast<T *>(out.ptr() + x), finalize_quantization<T>(in_s32, result_shift_s32, min, max));
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                int in_value = *(reinterpret_cast<const int *>(in.ptr()) + x);

                // Quantize
                in_value = ((in_value + _output_stage->gemmlowp_offset) * _output_stage->gemmlowp_multiplier) >> _output_stage->gemmlowp_shift;

                // Store the result
                *(out.ptr() + x) = static_cast<T>(utility::clamp<int>(in_value, clamp_min, clamp_max));
            }
        },
        in, out);
    }
}

NEGEMMLowpQuantizeDownInt32ScaleKernel::NEGEMMLowpQuantizeDownInt32ScaleKernel()
    : _func(nullptr), _input(nullptr), _bias(nullptr), _output(nullptr), _output_stage(nullptr), _is_bounded_relu(false)
{
}

void NEGEMMLowpQuantizeDownInt32ScaleKernel::configure(const ITensor *input, const ITensor *bias, ITensor *output, const GEMMLowpOutputStageInfo *output_stage)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, output_stage);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->clone()->set_data_type(output_stage->output_data_type));

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(),
                                                  (bias != nullptr) ? bias->info() : nullptr,
                                                  output->info(),
                                                  output_stage));

    _input        = input;
    _bias         = bias;
    _output       = output;
    _output_stage = output_stage;

    // Configure kernel window
    Window      win = calculate_max_window(*input->info(), Steps());
    Coordinates coord;
    coord.set_num_dimensions(output->info()->num_dimensions());
    output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));

    INEKernel::configure(win);

    // Check if we need to clamp the result using min and max
    _is_bounded_relu = ((_output_stage->gemmlowp_min_bound != _output_stage->gemmlowp_max_bound)
                        && !(_output_stage->gemmlowp_min_bound == std::get<0>(quantization::get_min_max_values_from_quantized_data_type(output_stage->output_data_type))
                             && _output_stage->gemmlowp_max_bound == std::get<1>(quantization::get_min_max_values_from_quantized_data_type(output_stage->output_data_type))));
    if(_output_stage->output_data_type == DataType::QASYMM8)
    {
        _func = &NEGEMMLowpQuantizeDownInt32ScaleKernel::run<uint8_t>;
    }
    else if(_output_stage->output_data_type == DataType::QASYMM8_SIGNED)
    {
        _func = &NEGEMMLowpQuantizeDownInt32ScaleKernel::run<int8_t>;
    }
    else
    {
        ARM_COMPUTE_ERROR("Data type not supported");
    }
}

Status NEGEMMLowpQuantizeDownInt32ScaleKernel::validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, const GEMMLowpOutputStageInfo *output_stage)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, bias, output, output_stage));

    return Status{};
}

void NEGEMMLowpQuantizeDownInt32ScaleKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    (this->*_func)(window);
}
} // namespace arm_compute