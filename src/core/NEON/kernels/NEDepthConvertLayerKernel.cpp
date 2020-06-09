/*
 * Copyright (c) 2016-2020 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEDepthConvertLayerKernel.h"

#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/NEON/NEMath.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/SaturateCast.h"

#include "arm_compute/core/NEON/wrapper/wrapper.h"

using namespace arm_compute;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, ConvertPolicy policy, uint32_t shift)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(output);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_BF16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_BF16_UNSUPPORTED(output);
    ARM_COMPUTE_UNUSED(policy);
    ARM_COMPUTE_RETURN_ERROR_ON(input == output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8_SIGNED, DataType::QASYMM8, DataType::U8,
                                                         DataType::S16, DataType::U16, DataType::BFLOAT16, DataType::F16,
                                                         DataType::F32, DataType::S32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::QASYMM8_SIGNED, DataType::QASYMM8, DataType::U8,
                                                         DataType::S16, DataType::U16, DataType::BFLOAT16, DataType::F16,
                                                         DataType::U32, DataType::S32, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(shift >= 8);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::QASYMM8_SIGNED && (output->data_type() != DataType::S16 && output->data_type() != DataType::S32
                                                                                       && output->data_type() != DataType::F16 && output->data_type() != DataType::F32),
                                    "Only data_types supported [in] QASYMM8 -> [out] U16, S16, S32, F16, F32");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::QASYMM8 && (output->data_type() != DataType::S16 && output->data_type() != DataType::U16
                                                                                && output->data_type() != DataType::S32 && output->data_type() != DataType::F16 && output->data_type() != DataType::F32),
                                    "Only data_types supported [in] QASYMM8 -> [out] U16, S16, S32, F16, F32");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::U8 && (output->data_type() != DataType::S16 && output->data_type() != DataType::U16
                                                                           && output->data_type() != DataType::S32 && output->data_type() != DataType::F16 && output->data_type() != DataType::F32),
                                    "Only data_types supported [in] U8 -> [out] U16, S16, S32, F16, F32");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::U16 && (output->data_type() != DataType::U8 && output->data_type() != DataType::U32),
                                    "Only data_types supported [in] U16 ->  [out] U8, U32");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::S16 && (output->data_type() != DataType::QASYMM8_SIGNED && output->data_type() != DataType::U8 && output->data_type() != DataType::S32),
                                    "Only data_types supported [in] S16 ->  [out] U8, S32");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::BFLOAT16 && output->data_type() != DataType::F32,
                                    "Only data_types supported [in] BFLOAT16 ->  [out] F32");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::F16 && (output->data_type() != DataType::QASYMM8_SIGNED && output->data_type() != DataType::QASYMM8
                                                                            && output->data_type() != DataType::U8
                                                                            && output->data_type() != DataType::F32 && output->data_type() != DataType::S32),
                                    "Only data_types supported [in] F16 ->  [out] QASYMM8, F32, S32, U8");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::F32 && (output->data_type() != DataType::QASYMM8_SIGNED && output->data_type() != DataType::QASYMM8
                                                                            && output->data_type() != DataType::F16 && output->data_type() != DataType::BFLOAT16
                                                                            && output->data_type() != DataType::S32 && output->data_type() != DataType::U8),
                                    "Only data_types supported [in] F32 ->  [out] QASYMM8, BFLOAT16, F16, S32, U8");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::S32 && (output->data_type() != DataType::QASYMM8_SIGNED && output->data_type() != DataType::QASYMM8
                                                                            && output->data_type() != DataType::F16
                                                                            && output->data_type() != DataType::F32 && output->data_type() != DataType::U8),
                                    "Only data_types supported [in] S32 ->  [out] QASYMM8, F16, F32, U8");

    // Validate in case of configured output
    if(output->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
    }

    return Status{};
}
} // namespace

NEDepthConvertLayerKernel::NEDepthConvertLayerKernel()
    : _input(nullptr), _output(nullptr), _policy(), _shift(0)
{
}

void NEDepthConvertLayerKernel::configure(const ITensor *input, ITensor *output, ConvertPolicy policy, uint32_t shift)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Auto initialize output shape if not initialized (We can only auto-configure the shape, datatype must be given)
    set_shape_if_empty(*output->info(), input->info()->tensor_shape());

    _input  = input;
    _output = output;
    _policy = policy;
    _shift  = shift;

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), policy, shift));

    // Configure kernel window
    Window      win = calculate_max_window(*input->info(), Steps());
    Coordinates coord;
    coord.set_num_dimensions(output->info()->num_dimensions());
    output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));

    ICPPKernel::configure(win);
}

Status NEDepthConvertLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, ConvertPolicy policy, uint32_t shift)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, policy, shift));
    return Status{};
}

void NEDepthConvertLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_input, _output);
    ARM_COMPUTE_ERROR_ON(_input == _output);

    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());
    const int  window_step_x  = 16;

    Window win{ window };
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(_input, win);
    Iterator output(_output, win);

    switch(_input->info()->data_type())
    {
        case DataType::QASYMM8_SIGNED:
        {
            const int16x8_t b = vdupq_n_s16(_shift);

            switch(_output->info()->data_type())
            {
                case DataType::S16:
                {
                    /* Up-conversion QASYMM8_SIGNED -> S16 */
                    execute_window_loop(win, [&](const Coordinates &)
                    {
                        const auto input_ptr  = reinterpret_cast<const int8_t *>(input.ptr());
                        const auto output_ptr = reinterpret_cast<int16_t *>(output.ptr());
                        int        x          = window_start_x;

                        for(; x <= (window_end_x - window_step_x); x += window_step_x)
                        {
                            const int8x16_t texels_s8 = vld1q_s8(input_ptr + x);

                            const int16x8x2_t texels =
                            {
                                {
                                    vshlq_s16(vmovl_s8(vget_low_s8(texels_s8)), b),
                                    vshlq_s16(vmovl_s8(vget_high_s8(texels_s8)), b)
                                }
                            };

                            vst1q_s16(output_ptr + x, texels.val[0]);
                            vst1q_s16(output_ptr + x + 8, texels.val[1]);
                        }

                        // Compute left-over elements
                        for(; x < window_end_x; ++x)
                        {
                            *(output_ptr + x) = static_cast<int16_t>(*(input_ptr + x) << _shift);
                        }
                    },
                    input, output);
                    break;
                }
                case DataType::S32:
                {
                    /* Up-conversion QASYMM8_SIGNED -> S32 */
                    execute_window_loop(win, [&](const Coordinates &)
                    {
                        const auto input_ptr  = reinterpret_cast<const int8_t *>(input.ptr());
                        const auto output_ptr = reinterpret_cast<int32_t *>(output.ptr());
                        int        x          = window_start_x;

                        for(; x <= (window_end_x - window_step_x); x += window_step_x)
                        {
                            const int8x16_t texels_s8 = vld1q_s8(input_ptr + x);

                            const int16x8x2_t texels =
                            {
                                {
                                    vshlq_s16(vmovl_s8(vget_low_s8(texels_s8)), b),
                                    vshlq_s16(vmovl_s8(vget_high_s8(texels_s8)), b)
                                }
                            };

                            vst1q_s32(output_ptr + x, vmovl_s16(vget_low_s16(texels.val[0])));
                            vst1q_s32(output_ptr + x + 4, vmovl_s16(vget_high_s16(texels.val[0])));
                            vst1q_s32(output_ptr + x + 8, vmovl_s16(vget_low_s16(texels.val[1])));
                            vst1q_s32(output_ptr + x + 12, vmovl_s16(vget_high_s16(texels.val[1])));
                        }

                        // Compute left-over elements
                        for(; x < window_end_x; ++x)
                        {
                            *(output_ptr + x) = static_cast<int32_t>(*(input_ptr + x) << _shift);
                        }
                    },
                    input, output);
                    break;
                }
                case DataType::F32:
                {
                    /* Up-conversion QASYMM8_SIGNED -> F32 */
                    execute_window_loop(win, [&](const Coordinates &)
                    {
                        const auto input_ptr  = reinterpret_cast<const int8_t *>(input.ptr());
                        const auto output_ptr = reinterpret_cast<float *>(output.ptr());

                        int x = window_start_x;
                        for(; x <= (window_end_x - window_step_x); x += window_step_x)
                        {
                            const int8x16_t texels_s8 = vld1q_s8(reinterpret_cast<int8_t *>(input.ptr()));

                            const int16x8x2_t texels =
                            {
                                {
                                    vshlq_s16(vmovl_s8(vget_low_s8(texels_s8)), b),
                                    vshlq_s16(vmovl_s8(vget_high_s8(texels_s8)), b)
                                }
                            };
                            vst1q_f32(output_ptr + x, vcvtq_f32_s32(vmovl_s16(vget_low_s16(texels.val[0]))));
                            vst1q_f32(output_ptr + x + 4, vcvtq_f32_s32(vmovl_s16(vget_high_s16(texels.val[0]))));
                            vst1q_f32(output_ptr + x + 8, vcvtq_f32_s32(vmovl_s16(vget_low_s16(texels.val[1]))));
                            vst1q_f32(output_ptr + x + 12, vcvtq_f32_s32(vmovl_s16(vget_high_s16(texels.val[1]))));
                        }

                        // Compute left-over elements
                        for(; x < window_end_x; ++x)
                        {
                            *(output_ptr + x) = static_cast<float>(*(input_ptr + x) << _shift);
                        }
                    },
                    input, output);
                    break;
                }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                {
                    /* Up-conversion QASYMM8_SIGNED -> F16 */
                    execute_window_loop(win, [&](const Coordinates &)
                    {
                        const auto input_ptr  = reinterpret_cast<const int8_t *>(input.ptr());
                        const auto output_ptr = reinterpret_cast<float16_t *>(output.ptr());
                        int        x          = window_start_x;

                        for(; x <= (window_end_x - window_step_x); x += window_step_x)
                        {
                            const int8x16_t texels_s8 = vld1q_s8(input_ptr + x);

                            const int16x8x2_t texels =
                            {
                                {
                                    vshlq_s16(vmovl_s8(vget_low_s8(texels_s8)), b),
                                    vshlq_s16(vmovl_s8(vget_high_s8(texels_s8)), b)
                                }
                            };
                            vst1q_f16(output_ptr + x, vcvtq_f16_s16(texels.val[0]));
                            vst1q_f16(output_ptr + x + 8, vcvtq_f16_s16(texels.val[1]));
                        }

                        // Compute left-over elements
                        for(; x < window_end_x; ++x)
                        {
                            *(output_ptr + x) = static_cast<float16_t>(*(input_ptr + x) << _shift);
                        }
                    },
                    input, output);
                    break;
                }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

                default:
                    ARM_COMPUTE_ERROR("Output data type not supported");
            }
            break;
        }

        case DataType::QASYMM8:
        case DataType::U8:
        {
            const int16x8_t b = vdupq_n_s16(_shift);

            switch(_output->info()->data_type())
            {
                case DataType::S16:
                {
                    /* Up-conversion U8 -> S16 */
                    execute_window_loop(win, [&](const Coordinates &)
                    {
                        const auto input_ptr  = reinterpret_cast<const uint8_t *>(input.ptr());
                        const auto output_ptr = reinterpret_cast<int16_t *>(output.ptr());

                        int x = window_start_x;
                        for(; x <= (window_end_x - window_step_x); x += window_step_x)
                        {
                            const uint8x16_t texels_u8 = vld1q_u8(input_ptr + x);

                            const int16x8x2_t texels =
                            {
                                {
                                    vshlq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(texels_u8))), b),
                                    vshlq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(texels_u8))), b)
                                }
                            };

                            vst1q_s16(output_ptr + x, texels.val[0]);
                            vst1q_s16(output_ptr + x + 8, texels.val[1]);
                        }

                        // Compute left-over elements
                        for(; x < window_end_x; ++x)
                        {
                            auto in           = static_cast<int32_t>(*(input_ptr + x));
                            *(output_ptr + x) = in << _shift;
                        }
                    },
                    input, output);
                    break;
                }
                case DataType::S32:
                {
                    /* Up-conversion U8 -> S32 */
                    execute_window_loop(win, [&](const Coordinates &)
                    {
                        const auto input_ptr  = reinterpret_cast<const uint8_t *>(input.ptr());
                        const auto output_ptr = reinterpret_cast<int32_t *>(output.ptr());

                        int x = window_start_x;
                        for(; x <= (window_end_x - window_step_x); x += window_step_x)
                        {
                            const uint8x16_t texels_u8 = vld1q_u8(input_ptr + x);

                            const int16x8x2_t texels =
                            {
                                {
                                    vshlq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(texels_u8))), b),
                                    vshlq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(texels_u8))), b)
                                }
                            };

                            vst1q_s32(output_ptr + x, vmovl_s16(vget_low_s16(texels.val[0])));
                            vst1q_s32(output_ptr + x + 4, vmovl_s16(vget_high_s16(texels.val[0])));
                            vst1q_s32(output_ptr + x + 8, vmovl_s16(vget_low_s16(texels.val[1])));
                            vst1q_s32(output_ptr + x + 12, vmovl_s16(vget_high_s16(texels.val[1])));
                        }

                        // Compute left-over elements
                        for(; x < window_end_x; ++x)
                        {
                            auto in           = static_cast<uint32_t>(*(input_ptr + x));
                            *(output_ptr + x) = in << _shift;
                        }
                    },
                    input, output);
                    break;
                }
                case DataType::F32:
                {
                    /* Up-conversion U8 -> F32 */
                    execute_window_loop(win, [&](const Coordinates &)
                    {
                        const auto input_ptr  = reinterpret_cast<const uint8_t *>(input.ptr());
                        const auto output_ptr = reinterpret_cast<float *>(output.ptr());

                        int x = window_start_x;
                        for(; x <= (window_end_x - window_step_x); x += window_step_x)
                        {
                            const uint8x16_t texels_u8 = vld1q_u8(input_ptr + x);

                            const int16x8x2_t texels =
                            {
                                {
                                    vshlq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(texels_u8))), b),
                                    vshlq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(texels_u8))), b)
                                }
                            };
                            vst1q_f32(output_ptr + x, vcvtq_f32_s32(vmovl_s16(vget_low_s16(texels.val[0]))));
                            vst1q_f32(output_ptr + x + 4, vcvtq_f32_s32(vmovl_s16(vget_high_s16(texels.val[0]))));
                            vst1q_f32(output_ptr + x + 8, vcvtq_f32_s32(vmovl_s16(vget_low_s16(texels.val[1]))));
                            vst1q_f32(output_ptr + x + 12, vcvtq_f32_s32(vmovl_s16(vget_high_s16(texels.val[1]))));
                        }

                        // Compute left-over elements
                        for(; x < window_end_x; ++x)
                        {
                            auto in           = static_cast<uint32_t>(*(input_ptr + x));
                            *(output_ptr + x) = static_cast<float>(in << _shift);
                        }
                    },
                    input, output);
                    break;
                }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                {
                    /* Up-conversion U8 -> F16 */
                    execute_window_loop(win, [&](const Coordinates &)
                    {
                        const auto input_ptr  = reinterpret_cast<const uint8_t *>(input.ptr());
                        const auto output_ptr = reinterpret_cast<float16_t *>(output.ptr());

                        int x = window_start_x;
                        for(; x <= (window_end_x - window_step_x); x += window_step_x)
                        {
                            const uint8x16_t texels_u8 = vld1q_u8(input_ptr + x);

                            const int16x8x2_t texels =
                            {
                                {
                                    vshlq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(texels_u8))), b),
                                    vshlq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(texels_u8))), b)
                                }
                            };
                            vst1q_f16(output_ptr + x, vcvtq_f16_s16(texels.val[0]));
                            vst1q_f16(output_ptr + x + 8, vcvtq_f16_s16(texels.val[1]));
                        }

                        // Compute left-over elements
                        for(; x < window_end_x; ++x)
                        {
                            *(output_ptr + x) = static_cast<float16_t>(*(input_ptr + x) << _shift);
                        }
                    },
                    input, output);
                    break;
                }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::U16:
                {
                    /* Up-conversion U8 -> U16 */
                    execute_window_loop(win, [&](const Coordinates &)
                    {
                        const auto input_ptr  = reinterpret_cast<const uint8_t *>(input.ptr());
                        const auto output_ptr = reinterpret_cast<uint16_t *>(output.ptr());

                        int x = window_start_x;
                        for(; x <= (window_end_x - window_step_x); x += window_step_x)
                        {
                            const uint8x16_t texels_u8 = vld1q_u8(input_ptr + x);

                            const uint16x8x2_t texels =
                            {
                                {
                                    vshlq_u16(vmovl_u8(vget_low_u8(texels_u8)), b),
                                    vshlq_u16(vmovl_u8(vget_high_u8(texels_u8)), b)
                                }
                            };

                            vst1q_u16(output_ptr + x, texels.val[0]);
                            vst1q_u16(output_ptr + x + 8, texels.val[1]);
                        }

                        // Compute left-over elements
                        for(; x < window_end_x; ++x)
                        {
                            *(output_ptr + x) = static_cast<uint16_t>(*(input_ptr + x)) << _shift;
                        }
                    },
                    input, output);
                    break;
                }
                default:
                    ARM_COMPUTE_ERROR("Output data type not supported");
            }
            break;
        }
        case DataType::S16:
        {
            switch(_output->info()->data_type())
            {
                case DataType::QASYMM8_SIGNED:
                {
                    const int16x8_t b = vdupq_n_s16(-static_cast<int16_t>(_shift));

                    /* Down-conversion S16 -> QASYMM8_SIGNED */
                    if(ConvertPolicy::SATURATE == _policy)
                    {
                        execute_window_loop(win, [&](const Coordinates &)
                        {
                            const auto input_ptr  = reinterpret_cast<const int16_t *>(input.ptr());
                            const auto output_ptr = reinterpret_cast<int8_t *>(output.ptr());

                            int x = window_start_x;
                            for(; x <= (window_end_x - window_step_x); x += window_step_x)
                            {
                                const int16x8x2_t texels =
                                {
                                    {
                                        vqshlq_s16(vld1q_s16(input_ptr + x), b),
                                        vqshlq_s16(vld1q_s16(input_ptr + x + 8), b)
                                    }
                                };

                                vst1q_s8(output_ptr + x, vcombine_s8(vqmovn_s16(texels.val[0]), vqmovn_s16(texels.val[1])));
                            }

                            // Compute left-over elements
                            for(; x < window_end_x; ++x)
                            {
                                *(output_ptr + x) = utils::cast::saturate_cast<int8_t>(*(input_ptr + x) >> _shift);
                            }
                        },
                        input, output);
                    }
                    else
                    {
                        execute_window_loop(win, [&](const Coordinates &)
                        {
                            const auto input_ptr  = reinterpret_cast<const int16_t *>(input.ptr());
                            const auto output_ptr = reinterpret_cast<int8_t *>(output.ptr());

                            int x = window_start_x;
                            for(; x <= (window_end_x - window_step_x); x += window_step_x)
                            {
                                const int16x8x2_t texels =
                                {
                                    {
                                        vshlq_s16(vld1q_s16(input_ptr + x), b),
                                        vshlq_s16(vld1q_s16(input_ptr + x + 8), b)
                                    }
                                };

                                vst1q_s8(output_ptr + x, vcombine_s8(vmovn_s16(texels.val[0]), vmovn_s16(texels.val[1])));
                            }

                            // Compute left-over elements
                            for(; x < window_end_x; ++x)
                            {
                                *(output_ptr + x) = static_cast<int8_t>(*(input_ptr + x) >> _shift);
                            }
                        },
                        input, output);
                    }
                    break;
                }
                case DataType::U8:
                {
                    const int16x8_t b = vdupq_n_s16(-static_cast<int16_t>(_shift));

                    /* Down-conversion S16 -> U8 */
                    if(ConvertPolicy::SATURATE == _policy)
                    {
                        execute_window_loop(win, [&](const Coordinates &)
                        {
                            const auto input_ptr  = reinterpret_cast<const int16_t *>(input.ptr());
                            const auto output_ptr = reinterpret_cast<uint8_t *>(output.ptr());

                            int x = window_start_x;
                            for(; x <= (window_end_x - window_step_x); x += window_step_x)
                            {
                                const int16x8x2_t texels =
                                {
                                    {
                                        vqshlq_s16(vld1q_s16(input_ptr + x), b),
                                        vqshlq_s16(vld1q_s16(input_ptr + x + 8), b)
                                    }
                                };

                                vst1q_u8(output_ptr + x, vcombine_u8(vqmovun_s16(texels.val[0]), vqmovun_s16(texels.val[1])));
                            }

                            // Compute left-over elements
                            for(; x < window_end_x; ++x)
                            {
                                *(output_ptr + x) = utils::cast::saturate_cast<uint8_t>(*(input_ptr + x) >> _shift);
                            }
                        },
                        input, output);
                    }
                    else
                    {
                        execute_window_loop(win, [&](const Coordinates &)
                        {
                            const auto input_ptr  = reinterpret_cast<const int16_t *>(input.ptr());
                            const auto output_ptr = reinterpret_cast<uint8_t *>(output.ptr());

                            int x = window_start_x;
                            for(; x <= (window_end_x - window_step_x); x += window_step_x)
                            {
                                const int16x8x2_t texels =
                                {
                                    {
                                        vshlq_s16(vld1q_s16(input_ptr + x), b),
                                        vshlq_s16(vld1q_s16(input_ptr + x + 8), b)
                                    }
                                };

                                vst1q_u8(output_ptr + x, vcombine_u8(vmovn_u16(vreinterpretq_u16_s16(texels.val[0])),
                                                                     vmovn_u16(vreinterpretq_u16_s16(texels.val[1]))));
                            }

                            // Compute left-over elements
                            for(; x < window_end_x; ++x)
                            {
                                *(output_ptr + x) = static_cast<uint8_t>(*(input_ptr + x) >> _shift);
                            }
                        },
                        input, output);
                    }
                    break;
                }
                case DataType::S32:
                {
                    const int32x4_t b = vdupq_n_s32(_shift);

                    /* Up-conversion S16 -> S32 */
                    execute_window_loop(win, [&](const Coordinates &)
                    {
                        const auto input_ptr  = reinterpret_cast<const int16_t *>(input.ptr());
                        const auto output_ptr = reinterpret_cast<int32_t *>(output.ptr());

                        int x = window_start_x;
                        for(; x <= (window_end_x - window_step_x); x += window_step_x)
                        {
                            const int16x8x2_t texels =
                            {
                                {
                                    vld1q_s16(input_ptr + x),
                                    vld1q_s16(input_ptr + x + 8)
                                }
                            };

                            const int32x4x4_t texels_s32 =
                            {
                                {
                                    vshlq_s32(vmovl_s16(vget_low_s16(texels.val[0])), b),
                                    vshlq_s32(vmovl_s16(vget_high_s16(texels.val[0])), b),
                                    vshlq_s32(vmovl_s16(vget_low_s16(texels.val[1])), b),
                                    vshlq_s32(vmovl_s16(vget_high_s16(texels.val[1])), b)
                                }
                            };

                            vst1q_s32(output_ptr + x, texels_s32.val[0]);
                            vst1q_s32(output_ptr + x + 4, texels_s32.val[1]);
                            vst1q_s32(output_ptr + x + 8, texels_s32.val[2]);
                            vst1q_s32(output_ptr + x + 12, texels_s32.val[3]);
                        }

                        // Compute left-over elements
                        for(; x < window_end_x; ++x)
                        {
                            *(output_ptr + x) = static_cast<int32_t>(*(input_ptr + x) << _shift);
                        }
                    },
                    input, output);
                    break;
                }
                default:
                    ARM_COMPUTE_ERROR("Output data type not supported");
            }
            break;
        }
        case DataType::U16:
        {
            switch(_output->info()->data_type())
            {
                case DataType::U8:
                {
                    const int16x8_t b = vdupq_n_s16(-static_cast<int16_t>(_shift));

                    /* Down-conversion U16 -> U8 */
                    if(ConvertPolicy::SATURATE == _policy)
                    {
                        execute_window_loop(win, [&](const Coordinates &)
                        {
                            const auto input_ptr  = reinterpret_cast<const uint16_t *>(input.ptr());
                            const auto output_ptr = reinterpret_cast<uint8_t *>(output.ptr());

                            int x = window_start_x;
                            for(; x <= (window_end_x - window_step_x); x += window_step_x)
                            {
                                const uint16x8x2_t texels =
                                {
                                    {
                                        vqshlq_u16(vld1q_u16(input_ptr + x), b),
                                        vqshlq_u16(vld1q_u16(input_ptr + x + 8), b)
                                    }
                                };

                                vst1q_u8(output_ptr + x, vcombine_u8(vqmovn_u16(texels.val[0]), vqmovn_u16(texels.val[1])));
                            }

                            // Compute left-over elements
                            for(; x < window_end_x; ++x)
                            {
                                *(output_ptr + x) = utils::cast::saturate_cast<uint8_t>(*(input_ptr + x) >> _shift);
                            }
                        },
                        input, output);
                    }
                    else
                    {
                        execute_window_loop(win, [&](const Coordinates &)
                        {
                            const auto input_ptr  = reinterpret_cast<const uint16_t *>(input.ptr());
                            const auto output_ptr = reinterpret_cast<uint8_t *>(output.ptr());

                            int x = window_start_x;
                            for(; x <= (window_end_x - window_step_x); x += window_step_x)
                            {
                                const uint16x8x2_t texels =
                                {
                                    {
                                        vshlq_u16(vld1q_u16(input_ptr + x), b),
                                        vshlq_u16(vld1q_u16(input_ptr + x + 8), b)
                                    }
                                };

                                vst1q_u8(output_ptr + x, vcombine_u8(vmovn_u16(texels.val[0]), vmovn_u16(texels.val[1])));
                            }

                            // Compute left-over elements
                            for(; x < window_end_x; ++x)
                            {
                                *(output_ptr + x) = static_cast<uint8_t>(*(input_ptr + x) >> _shift);
                            }

                        },
                        input, output);
                    }
                    break;
                }
                case DataType::U32:
                {
                    const int32x4_t b = vdupq_n_s32(_shift);

                    /* Up-conversion U16 -> U32 */
                    execute_window_loop(win, [&](const Coordinates &)
                    {
                        const auto input_ptr  = reinterpret_cast<const uint16_t *>(input.ptr());
                        const auto output_ptr = reinterpret_cast<uint32_t *>(output.ptr());

                        int x = window_start_x;
                        for(; x <= (window_end_x - window_step_x); x += window_step_x)
                        {
                            const uint16x8x2_t texels =
                            {
                                {
                                    vld1q_u16(input_ptr + x),
                                    vld1q_u16(input_ptr + x + 8)
                                }
                            };

                            vst1q_u32(output_ptr + x, vshlq_u32(vmovl_u16(vget_low_u16(texels.val[0])), b));
                            vst1q_u32(output_ptr + x + 4, vshlq_u32(vmovl_u16(vget_high_u16(texels.val[0])), b));
                            vst1q_u32(output_ptr + x + 8, vshlq_u32(vmovl_u16(vget_low_u16(texels.val[1])), b));
                            vst1q_u32(output_ptr + x + 12, vshlq_u32(vmovl_u16(vget_high_u16(texels.val[1])), b));
                        }
                        // Compute left-over elements
                        for(; x < window_end_x; ++x)
                        {
                            *(output_ptr + x) = static_cast<uint32_t>(*(input_ptr + x) << _shift);
                        }

                    },
                    input, output);
                    break;
                }
                default:
                    ARM_COMPUTE_ERROR("Output data type not supported");
            }
            break;
        }
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(ARM_COMPUTE_FORCE_BF16)
        case DataType::BFLOAT16:
            switch(_output->info()->data_type())
            {
                case DataType::F32:
                {
                    /* Up-conversion BFLOAT16 -> F32 */
                    execute_window_loop(win, [&](const Coordinates &)
                    {
                        const auto input_ptr  = reinterpret_cast<const bfloat16 *>(input.ptr());
                        const auto output_ptr = reinterpret_cast<float *>(output.ptr());

                        int x = window_start_x;
                        for(; x <= (window_end_x - window_step_x); x += window_step_x)
                        {
                            const uint16x8x2_t texels =
                            {
                                {
                                    vld1q_u16(reinterpret_cast<uint16_t *>(input.ptr())),
                                    vld1q_u16(reinterpret_cast<uint16_t *>(input.ptr()) + 8)
                                }
                            };

                            vst1q_f32(reinterpret_cast<float *>(output.ptr()),
                                      vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_low_u16(texels.val[0])), 16)));
                            vst1q_f32(reinterpret_cast<float *>(output.ptr()) + 4,
                                      vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_high_u16(texels.val[0])), 16)));
                            vst1q_f32(reinterpret_cast<float *>(output.ptr()) + 8,
                                      vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_low_u16(texels.val[1])), 16)));
                            vst1q_f32(reinterpret_cast<float *>(output.ptr()) + 12,
                                      vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_high_u16(texels.val[1])), 16)));
                        }

                        for(; x < window_end_x; ++x)
                        {
                            *(output_ptr + x) = float(*(input_ptr + x));
                        }
                    },
                    input, output);
                    break;
                }
                default:
                    ARM_COMPUTE_ERROR("Output data type unsupported");
            }
            break;
#endif /* defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(ARM_COMPUTE_FORCE_BF16) */
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            switch(_output->info()->data_type())
            {
                case DataType::QASYMM8_SIGNED:
                {
                    const float16_t   scale_s = 1 << _shift;
                    const float16x8_t scale   = vdupq_n_f16(scale_s);

                    /* Down-conversion F16 -> QASYMM8_SIGNED (Always saturating) */
                    execute_window_loop(win, [&](const Coordinates &)
                    {
                        const auto input_ptr  = reinterpret_cast<const float16_t *>(input.ptr());
                        const auto output_ptr = reinterpret_cast<int8_t *>(output.ptr());

                        int x = window_start_x;
                        for(; x <= (window_end_x - window_step_x); x += window_step_x)
                        {
                            const float16x8x2_t texels =
                            {
                                {
                                    vmulq_f16(vld1q_f16(input_ptr + x), scale),
                                    vmulq_f16(vld1q_f16(input_ptr + x + 8), scale),
                                }
                            };

                            vst1q_s8(output_ptr + x, vcombine_s8(vqmovn_s16(vcvtq_s16_f16(texels.val[0])), vqmovn_s16(vcvtq_s16_f16(texels.val[1]))));
                        }

                        // Compute left-over elements
                        for(; x < window_end_x; ++x)
                        {
                            *(output_ptr + x) = utils::cast::saturate_cast<int8_t>(*(input_ptr + x) * scale_s);
                        }
                    },
                    input, output);
                    break;
                }
                case DataType::QASYMM8:
                case DataType::U8:
                {
                    const float16_t   scale_s = 1 << _shift;
                    const float16x8_t scale   = vdupq_n_f16(scale_s);

                    /* Down-conversion F16 -> QASYMM8/U8 (Always saturating) */
                    execute_window_loop(win, [&](const Coordinates &)
                    {
                        const auto input_ptr  = reinterpret_cast<const float16_t *>(input.ptr());
                        const auto output_ptr = reinterpret_cast<uint8_t *>(output.ptr());

                        int x = window_start_x;
                        for(; x <= (window_end_x - window_step_x); x += window_step_x)
                        {
                            const float16x8x2_t texels =
                            {
                                {
                                    vmulq_f16(vld1q_f16(input_ptr + x), scale),
                                    vmulq_f16(vld1q_f16(input_ptr + x + 8), scale),
                                }
                            };

                            vst1q_u8(output_ptr + x, vcombine_u8(vqmovun_s16(vcvtq_s16_f16(texels.val[0])), vqmovun_s16(vcvtq_s16_f16(texels.val[1]))));
                        }

                        // Compute left-over elements
                        for(; x < window_end_x; ++x)
                        {
                            *(output_ptr + x) = utils::cast::saturate_cast<uint8_t>(*(input_ptr + x) * scale_s);
                        }

                    },
                    input, output);
                    break;
                }
                case DataType::F32:
                {
                    const float       scale_s = 1 << _shift;
                    const float32x4_t scale   = vdupq_n_f32(scale_s);

                    /* Up-conversion F16 -> F32 */
                    execute_window_loop(win, [&](const Coordinates &)
                    {
                        const auto input_ptr  = reinterpret_cast<const float16_t *>(input.ptr());
                        const auto output_ptr = reinterpret_cast<float *>(output.ptr());

                        int x = window_start_x;
                        for(; x <= (window_end_x - window_step_x); x += window_step_x)
                        {
                            const float16x8x2_t texels =
                            {
                                {
                                    vld1q_f16(input_ptr + x),
                                    vld1q_f16(input_ptr + x + 8)
                                }
                            };
                            vst1q_f32(output_ptr + x, vmulq_f32(vcvt_f32_f16(vget_low_f16(texels.val[0])), scale));
                            vst1q_f32(output_ptr + x + 4, vmulq_f32(vcvt_f32_f16(vget_high_f16(texels.val[0])), scale));
                            vst1q_f32(output_ptr + x + 8, vmulq_f32(vcvt_f32_f16(vget_low_f16(texels.val[1])), scale));
                            vst1q_f32(output_ptr + x + 12, vmulq_f32(vcvt_f32_f16(vget_high_f16(texels.val[1])), scale));
                        }

                        // Compute left-over elements
                        for(; x < window_end_x; ++x)
                        {
                            *(output_ptr + x) = static_cast<float>(*(input_ptr + x) * scale_s);
                        }
                    },
                    input, output);
                    break;
                }
                case DataType::S32:
                {
                    const float       scale_s = 1 << _shift;
                    const float32x4_t scale   = vdupq_n_f32(scale_s);

                    /* Up-conversion F16 -> S32 */
                    execute_window_loop(win, [&](const Coordinates &)
                    {
                        const auto input_ptr  = reinterpret_cast<const float16_t *>(input.ptr());
                        const auto output_ptr = reinterpret_cast<int32_t *>(output.ptr());

                        int x = window_start_x;
                        for(; x <= (window_end_x - window_step_x); x += window_step_x)
                        {
                            const float16x8x2_t texels =
                            {
                                {
                                    vld1q_f16(input_ptr + x),
                                    vld1q_f16(input_ptr + x + 8)
                                }
                            };

                            vst1q_s32(output_ptr + x, vcvtq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_low_f16(texels.val[0])), scale)));
                            vst1q_s32(output_ptr + x + 4, vcvtq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_high_f16(texels.val[0])), scale)));
                            vst1q_s32(output_ptr + x + 8, vcvtq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_low_f16(texels.val[1])), scale)));
                            vst1q_s32(output_ptr + x + 12, vcvtq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_high_f16(texels.val[1])), scale)));
                        }

                        // Compute left-over elements
                        for(; x < window_end_x; ++x)
                        {
                            *(output_ptr + x) = static_cast<int32_t>(*(input_ptr + x) * scale_s);
                        }
                    },
                    input, output);
                    break;
                }
                default:
                    ARM_COMPUTE_ERROR("Output data type not supported");
            }
            break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        case DataType::F32:
            switch(_output->info()->data_type())
            {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                {
                    const float       scale_s = 1.f / (1 << _shift);
                    const float32x4_t scale   = vdupq_n_f32(scale_s);

                    /* Down-conversion F32 -> F16 */
                    execute_window_loop(win, [&](const Coordinates &)
                    {
                        const auto input_ptr  = reinterpret_cast<const float *>(input.ptr());
                        const auto output_ptr = reinterpret_cast<float16_t *>(output.ptr());

                        int x = window_start_x;
                        for(; x <= (window_end_x - window_step_x); x += window_step_x)
                        {
                            const float32x4x4_t texels =
                            {
                                {
                                    vmulq_f32(vld1q_f32(input_ptr + x), scale),
                                    vmulq_f32(vld1q_f32(input_ptr + x + 4), scale),
                                    vmulq_f32(vld1q_f32(input_ptr + x + 8), scale),
                                    vmulq_f32(vld1q_f32(input_ptr + x + 12), scale)
                                }
                            };

                            vst1q_f16(output_ptr + x, vcombine_f16(vcvt_f16_f32(texels.val[0]), vcvt_f16_f32(texels.val[1])));
                            vst1q_f16(output_ptr + x + 8, vcombine_f16(vcvt_f16_f32(texels.val[2]), vcvt_f16_f32(texels.val[3])));
                        }

                        // Compute left-over elements
                        for(; x < window_end_x; ++x)
                        {
                            *(output_ptr + x) = static_cast<float16_t>(*(input_ptr + x) * scale_s);
                        }
                    },
                    input, output);
                    break;
                }
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(ARM_COMPUTE_FORCE_BF16)
                case DataType::BFLOAT16:
                {
                    /* Down-conversion F32 -> BFLOAT16 */
                    execute_window_loop(win, [&](const Coordinates &)
                    {
                        const auto input_ptr  = reinterpret_cast<const float *>(input.ptr());
                        const auto output_ptr = reinterpret_cast<bfloat16 *>(output.ptr());

                        int x = window_start_x;
                        for(; x <= (window_end_x - window_step_x); x += window_step_x)
                        {
                            wrapper::vcvt_bf16_f32(reinterpret_cast<float *>(input.ptr()),
                                                   reinterpret_cast<uint16_t *>(output.ptr()));
                            wrapper::vcvt_bf16_f32(reinterpret_cast<float *>(input.ptr()) + 8,
                                                   reinterpret_cast<uint16_t *>(output.ptr()) + 8);
                        }

                        for(; x < window_end_x; ++x)
                        {
                            *(output_ptr + x) = *(input_ptr + x);
                        }
                    },
                    input, output);
                    break;
                }
#endif /* defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(ARM_COMPUTE_FORCE_BF16) */
                case DataType::S32:
                {
                    const float       scale_s = 1.f / (1 << _shift);
                    const float32x4_t scale   = vdupq_n_f32(scale_s);

                    /* Conversion F32 -> S32 */
                    execute_window_loop(win, [&](const Coordinates &)
                    {
                        const auto input_ptr  = reinterpret_cast<const float *>(input.ptr());
                        const auto output_ptr = reinterpret_cast<int32_t *>(output.ptr());

                        int x = window_start_x;
                        for(; x <= (window_end_x - window_step_x); x += window_step_x)
                        {
                            const float32x4x4_t texels =
                            {
                                {
                                    vmulq_f32(vld1q_f32(input_ptr + x), scale),
                                    vmulq_f32(vld1q_f32(input_ptr + x + 4), scale),
                                    vmulq_f32(vld1q_f32(input_ptr + x + 8), scale),
                                    vmulq_f32(vld1q_f32(input_ptr + x + 12), scale),
                                }
                            };

                            vst1q_s32(output_ptr + x, vcvtq_s32_f32(texels.val[0]));
                            vst1q_s32(output_ptr + x + 4, vcvtq_s32_f32(texels.val[1]));
                            vst1q_s32(output_ptr + x + 8, vcvtq_s32_f32(texels.val[2]));
                            vst1q_s32(output_ptr + x + 12, vcvtq_s32_f32(texels.val[3]));
                        }

                        // Compute left-over elements
                        for(; x < window_end_x; ++x)
                        {
                            *(output_ptr + x) = static_cast<int32_t>(*(input_ptr + x) * scale_s);
                        }
                    },
                    input, output);
                    break;
                }
                case DataType::QASYMM8:
                case DataType::U8:
                {
                    const float       scale_s = 1.f / (1 << _shift);
                    const float32x4_t scale   = vdupq_n_f32(scale_s);

                    /* Down-conversion F32 -> U8 */
                    execute_window_loop(win, [&](const Coordinates &)
                    {
                        const auto input_ptr  = reinterpret_cast<const float *>(input.ptr());
                        const auto output_ptr = reinterpret_cast<uint8_t *>(output.ptr());

                        int x = window_start_x;
                        for(; x <= (window_end_x - window_step_x); x += window_step_x)
                        {
                            const float32x4x4_t texels =
                            {
                                {
                                    vmulq_f32(vld1q_f32(input_ptr + x), scale),
                                    vmulq_f32(vld1q_f32(input_ptr + x + 4), scale),
                                    vmulq_f32(vld1q_f32(input_ptr + x + 8), scale),
                                    vmulq_f32(vld1q_f32(input_ptr + x + 12), scale),
                                }
                            };

                            vst1_u8(output_ptr + x, vqmovn_u16(vcombine_u16(vqmovun_s32(vcvtq_s32_f32(texels.val[0])), vqmovun_s32(vcvtq_s32_f32(texels.val[1])))));
                            vst1_u8(output_ptr + x + 8, vqmovn_u16(vcombine_u16(vqmovun_s32(vcvtq_s32_f32(texels.val[2])), vqmovun_s32(vcvtq_s32_f32(texels.val[3])))));
                        }

                        // Compute left-over elements
                        for(; x < window_end_x; ++x)
                        {
                            *(output_ptr + x) = utils::cast::saturate_cast<uint8_t>(*(input_ptr + x) * scale_s);
                        }
                    },
                    input, output);
                    break;
                }
                case DataType::QASYMM8_SIGNED:
                {
                    const float       scale_s = 1.f / (1 << _shift);
                    const float32x4_t scale   = vdupq_n_f32(scale_s);

                    /* Down-conversion F32 -> QASYMM8_SIGNED */
                    execute_window_loop(win, [&](const Coordinates &)
                    {
                        const auto input_ptr  = reinterpret_cast<const float *>(input.ptr());
                        const auto output_ptr = reinterpret_cast<int8_t *>(output.ptr());

                        int x = window_start_x;
                        for(; x <= (window_end_x - window_step_x); x += window_step_x)
                        {
                            const float32x4x4_t texels =
                            {
                                {
                                    vmulq_f32(vld1q_f32(input_ptr + x), scale),
                                    vmulq_f32(vld1q_f32(input_ptr + x + 4), scale),
                                    vmulq_f32(vld1q_f32(input_ptr + x + 8), scale),
                                    vmulq_f32(vld1q_f32(input_ptr + x + 12), scale),
                                }
                            };

                            vst1_s8(output_ptr + x, vqmovn_s16(vcombine_s16(vqmovn_s32(vcvtq_s32_f32(texels.val[0])), vqmovn_s32(vcvtq_s32_f32(texels.val[1])))));
                            vst1_s8(output_ptr + x + 8, vqmovn_s16(vcombine_s16(vqmovn_s32(vcvtq_s32_f32(texels.val[2])), vqmovn_s32(vcvtq_s32_f32(texels.val[3])))));
                        }
                        // Compute left-over elements
                        for(; x < window_end_x; ++x)
                        {
                            *(output_ptr + x) = utils::cast::saturate_cast<int8_t>(*(input_ptr + x) * scale_s);
                        }
                    },
                    input, output);
                    break;
                }

                default:
                    ARM_COMPUTE_ERROR("Output data type not supported");
            }
            break;

        case DataType::S32:
            switch(_output->info()->data_type())
            {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                {
                    const float       scale_s = 1.f / (1 << _shift);
                    const float32x4_t scale   = vdupq_n_f32(scale_s);

                    /* Down-conversion S32 -> F16 */
                    execute_window_loop(win, [&](const Coordinates &)
                    {
                        const auto input_ptr  = reinterpret_cast<const int32_t *>(input.ptr());
                        const auto output_ptr = reinterpret_cast<float16_t *>(output.ptr());

                        int x = window_start_x;
                        for(; x <= (window_end_x - window_step_x); x += window_step_x)
                        {
                            const float32x4x4_t texels =
                            {
                                {
                                    vmulq_f32(vcvtq_f32_s32(vld1q_s32(input_ptr + x)), scale),
                                    vmulq_f32(vcvtq_f32_s32(vld1q_s32(input_ptr + x + 4)), scale),
                                    vmulq_f32(vcvtq_f32_s32(vld1q_s32(input_ptr + x + 8)), scale),
                                    vmulq_f32(vcvtq_f32_s32(vld1q_s32(input_ptr + x + 12)), scale)
                                }
                            };

                            vst1q_f16(output_ptr + x, vcombine_f16(vcvt_f16_f32(texels.val[0]), vcvt_f16_f32(texels.val[1])));
                            vst1q_f16(output_ptr + x + 8, vcombine_f16(vcvt_f16_f32(texels.val[2]), vcvt_f16_f32(texels.val[3])));
                        }

                        // Compute left-over elements
                        for(; x < window_end_x; ++x)
                        {
                            *(output_ptr + x) = static_cast<float16_t>(*(input_ptr + x) * scale_s);
                        }
                    },
                    input, output);
                    break;
                }
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
                case DataType::F32:
                {
                    const int       scale_s = 1.f / (1 << _shift);
                    const int32x4_t scale   = vdupq_n_s32(scale_s);

                    /* Conversion S32 -> F32 */
                    execute_window_loop(win, [&](const Coordinates &)
                    {
                        const auto input_ptr  = reinterpret_cast<const int32_t *>(input.ptr());
                        const auto output_ptr = reinterpret_cast<float *>(output.ptr());

                        int x = window_start_x;
                        for(; x <= (window_end_x - window_step_x); x += window_step_x)
                        {
                            const int32x4x4_t texels =
                            {
                                {
                                    vmulq_s32(vld1q_s32(input_ptr + x), scale),
                                    vmulq_s32(vld1q_s32(input_ptr + x + 4), scale),
                                    vmulq_s32(vld1q_s32(input_ptr + x + 8), scale),
                                    vmulq_s32(vld1q_s32(input_ptr + x + 12), scale),
                                }
                            };

                            vst1q_f32(output_ptr + x, vcvtq_f32_s32(texels.val[0]));
                            vst1q_f32(output_ptr + x + 4, vcvtq_f32_s32(texels.val[1]));
                            vst1q_f32(output_ptr + x + 8, vcvtq_f32_s32(texels.val[2]));
                            vst1q_f32(output_ptr + x + 12, vcvtq_f32_s32(texels.val[3]));
                        }

                        // Compute left-over elements
                        for(; x < window_end_x; ++x)
                        {
                            *(output_ptr + x) = static_cast<float>(*(input_ptr + x) * scale_s);
                        }
                    },
                    input, output);
                    break;
                }
                case DataType::QASYMM8_SIGNED:
                {
                    const int32x4_t b = vdupq_n_s32(-static_cast<int32_t>(_shift));

                    /* Down-conversion S32 -> QASYMM8_SIGNED */
                    if(ConvertPolicy::SATURATE == _policy)
                    {
                        execute_window_loop(win, [&](const Coordinates &)
                        {
                            const auto input_ptr  = reinterpret_cast<const int32_t *>(input.ptr());
                            const auto output_ptr = reinterpret_cast<int8_t *>(output.ptr());

                            int x = window_start_x;
                            for(; x <= (window_end_x - window_step_x); x += window_step_x)
                            {
                                const int32x4x4_t texels =
                                {
                                    {
                                        vqshlq_s32(vld1q_s32(input_ptr + x), b),
                                        vqshlq_s32(vld1q_s32(input_ptr + x + 4), b),
                                        vqshlq_s32(vld1q_s32(input_ptr + x + 8), b),
                                        vqshlq_s32(vld1q_s32(input_ptr + x + 12), b)
                                    }
                                };
                                vst1_s8(output_ptr + x, vqmovn_s16(vcombine_s16(vqmovn_s32(texels.val[0]), vqmovn_s32(texels.val[1]))));
                                vst1_s8(output_ptr + x + 8, vqmovn_s16(vcombine_s16(vqmovn_s32(texels.val[2]), vqmovn_s32(texels.val[3]))));
                            }

                            // Compute left-over elements
                            for(; x < window_end_x; ++x)
                            {
                                *(output_ptr + x) = utils::cast::saturate_cast<int8_t>(*(input_ptr + x) >> _shift);
                            }
                        },
                        input, output);
                    }
                    else
                    {
                        execute_window_loop(win, [&](const Coordinates &)
                        {
                            const auto input_ptr  = reinterpret_cast<const int32_t *>(input.ptr());
                            const auto output_ptr = reinterpret_cast<int8_t *>(output.ptr());

                            int x = window_start_x;
                            for(; x <= (window_end_x - window_step_x); x += window_step_x)
                            {
                                const int32x4x4_t texels =
                                {
                                    {
                                        vshlq_s32(vld1q_s32(input_ptr + x), b),
                                        vshlq_s32(vld1q_s32(input_ptr + x + 4), b),
                                        vshlq_s32(vld1q_s32(input_ptr + x + 8), b),
                                        vshlq_s32(vld1q_s32(input_ptr + x + 12), b)
                                    }
                                };

                                vst1_s8(output_ptr + x, vmovn_s16(vcombine_s16(vmovn_s32(texels.val[0]), vmovn_s32(texels.val[1]))));
                                vst1_s8(output_ptr + x + 8, vmovn_s16(vcombine_s16(vmovn_s32(texels.val[2]), vmovn_s32(texels.val[3]))));
                            }

                            // Compute left-over elements
                            for(; x < window_end_x; ++x)
                            {
                                *(output_ptr + x) = static_cast<int8_t>(*(input_ptr + x) >> _shift);
                            }
                        },
                        input, output);
                    }
                    break;
                }
                case DataType::QASYMM8:
                case DataType::U8:
                {
                    const int32x4_t b = vdupq_n_s32(-static_cast<int32_t>(_shift));

                    /* Down-conversion S32 -> U8 */
                    if(ConvertPolicy::SATURATE == _policy)
                    {
                        execute_window_loop(win, [&](const Coordinates &)
                        {
                            const auto input_ptr  = reinterpret_cast<const int32_t *>(input.ptr());
                            const auto output_ptr = reinterpret_cast<uint8_t *>(output.ptr());

                            int x = window_start_x;
                            for(; x <= (window_end_x - window_step_x); x += window_step_x)
                            {
                                const int32x4x4_t texels =
                                {
                                    {
                                        vqshlq_s32(vld1q_s32(input_ptr + x), b),
                                        vqshlq_s32(vld1q_s32(input_ptr + x + 4), b),
                                        vqshlq_s32(vld1q_s32(input_ptr + x + 8), b),
                                        vqshlq_s32(vld1q_s32(input_ptr + x + 12), b)
                                    }
                                };
                                vst1_u8(output_ptr + x, vqmovn_u16(vcombine_u16(vqmovun_s32(texels.val[0]), vqmovun_s32(texels.val[1]))));
                                vst1_u8(output_ptr + x + 8, vqmovn_u16(vcombine_u16(vqmovun_s32(texels.val[2]), vqmovun_s32(texels.val[3]))));
                            }

                            // Compute left-over elements
                            for(; x < window_end_x; ++x)
                            {
                                *(output_ptr + x) = utils::cast::saturate_cast<uint8_t>(*(input_ptr + x) >> _shift);
                            }
                        },
                        input, output);
                    }
                    else
                    {
                        execute_window_loop(win, [&](const Coordinates &)
                        {
                            const auto input_ptr  = reinterpret_cast<const int32_t *>(input.ptr());
                            const auto output_ptr = reinterpret_cast<uint8_t *>(output.ptr());

                            int x = window_start_x;
                            for(; x <= (window_end_x - window_step_x); x += window_step_x)
                            {
                                const int32x4x4_t texels =
                                {
                                    {
                                        vshlq_s32(vld1q_s32(input_ptr + x), b),
                                        vshlq_s32(vld1q_s32(input_ptr + x + 4), b),
                                        vshlq_s32(vld1q_s32(input_ptr + x + 8), b),
                                        vshlq_s32(vld1q_s32(input_ptr + x + 12), b)
                                    }
                                };

                                vst1_u8(output_ptr + x, vmovn_u16(vcombine_u16(vmovn_u32(vreinterpretq_u32_s32(texels.val[0])), vmovn_u32(vreinterpretq_u32_s32(texels.val[1])))));
                                vst1_u8(output_ptr + x + 8, vmovn_u16(vcombine_u16(vmovn_u32(vreinterpretq_u32_s32(texels.val[2])), vmovn_u32(vreinterpretq_u32_s32(texels.val[3])))));
                            }

                            // Compute left-over elements
                            for(; x < window_end_x; ++x)
                            {
                                *(output_ptr + x) = static_cast<uint8_t>(*(input_ptr + x) >> _shift);
                            }
                        },
                        input, output);
                    }
                    break;
                }
                default:
                    ARM_COMPUTE_ERROR("Output data type not supported");
            }
            break;
        default:
            ARM_COMPUTE_ERROR("Not supported");
    }
}
