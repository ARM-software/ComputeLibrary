/*
 * Copyright (c) 2016, 2017 ARM Limited.
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

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"

#include <arm_neon.h>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

NEDepthConvertLayerKernel::NEDepthConvertLayerKernel()
    : _input(nullptr), _output(nullptr), _policy(), _shift(0), _fixed_point_position_input(0), _fixed_point_position_output(0)
{
}

void NEDepthConvertLayerKernel::configure(ITensor *input, ITensor *output, ConvertPolicy policy, uint32_t shift)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::QS8, DataType::S16, DataType::U16, DataType::QS16, DataType::F32);

    _input  = input;
    _output = input;
    _policy = policy;
    _shift  = shift;

    if(output != nullptr)
    {
        // Auto initialize output shape if not initialized (We can only auto-configure the shape, datatype must be given)
        set_shape_if_empty(*output->info(), input->info()->tensor_shape());

        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8, DataType::QS8, DataType::S16, DataType::U16, DataType::QS16, DataType::U32, DataType::S32, DataType::F32);
        ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, output);

        // Set output
        _output = output;
    }

    // Set initial fixed point position of input and output
    _fixed_point_position_input  = input->info()->fixed_point_position();
    _fixed_point_position_output = _output->info()->fixed_point_position();

    // Set the fixed point position to the output tensor if needed
    if(is_data_type_fixed_point(input->info()->data_type()) && is_data_type_fixed_point(_output->info()->data_type()))
    {
        // If in-place set the fixed point position of the output tensor to be equal to shift
        _fixed_point_position_output = (_input == _output) ? static_cast<int>(_shift) : _fixed_point_position_output;
        // Set fixed point position to output tensor
        _output->info()->set_fixed_point_position(_fixed_point_position_output);
    }

    ARM_COMPUTE_ERROR_ON(shift >= 8 && (!is_data_type_fixed_point(input->info()->data_type()) && !is_data_type_fixed_point(output->info()->data_type())));
    ARM_COMPUTE_ERROR_ON(input == output && (data_size_from_type(input->info()->data_type()) != data_size_from_type(output->info()->data_type())));

    ARM_COMPUTE_ERROR_ON_MSG(input->info()->data_type() == DataType::U8 && (output->info()->data_type() != DataType::S16 && output->info()->data_type() != DataType::U16
                                                                            && output->info()->data_type() != DataType::S32),
                             "Only data_types supported [in] U8 -> [out] U16, S16, S32");

    ARM_COMPUTE_ERROR_ON_MSG(input->info()->data_type() == DataType::QS8 && (output->info()->data_type() != DataType::QS8 && output->info()->data_type() != DataType::F32),
                             "Only data_types supported [in] QS8 ->  [out] QS8, F32");

    ARM_COMPUTE_ERROR_ON_MSG(input->info()->data_type() == DataType::U16 && (output->info()->data_type() != DataType::U8 && output->info()->data_type() != DataType::U32),
                             "Only data_types supported [in] U16 ->  [out] U8, U32");

    ARM_COMPUTE_ERROR_ON_MSG(input->info()->data_type() == DataType::S16 && (output->info()->data_type() != DataType::U8 && output->info()->data_type() != DataType::S32),
                             "Only data_types supported [in] S16 ->  [out] U8, S32");

    ARM_COMPUTE_ERROR_ON_MSG(input->info()->data_type() == DataType::QS16 && (output->info()->data_type() != DataType::QS16 && output->info()->data_type() != DataType::F32),
                             "Only data_types supported [in] QS16 ->  [out] QS16, F32");

    ARM_COMPUTE_ERROR_ON_MSG(input->info()->data_type() == DataType::F32 && (output->info()->data_type() != DataType::QS8 && output->info()->data_type() != DataType::QS16),
                             "Only data_types supported [in] F32 ->  [out] QS8, QS16");

    constexpr unsigned int num_elems_processed_per_iteration = 16;

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(input->info(), 0, num_elems_processed_per_iteration);
    if(output != nullptr)
    {
        AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);
        update_window_and_padding(win, input_access, output_access);
        output_access.set_valid_region(win, input->info()->valid_region());
    }
    else
    {
        // In-place computation
        update_window_and_padding(win, input_access);
    }
    ICPPKernel::configure(win);
}

void NEDepthConvertLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(nullptr == _input);
    ARM_COMPUTE_ERROR_ON(nullptr == _output);
    ARM_COMPUTE_ERROR_ON(_input == _output);

    Iterator input(_input, window);
    Iterator output(_output, window);

    bool in_place = (_input == _output);

    switch(_input->info()->data_type())
    {
        case DataType::U8:
        {
            const int16x8_t b = vdupq_n_s16(_shift);

            switch(_output->info()->data_type())
            {
                case DataType::S16:
                {
                    /* Up-conversion U8 -> S16 */
                    execute_window_loop(window, [&](const Coordinates & id)
                    {
                        const uint8x16_t texels_u8 = vld1q_u8(input.ptr());

                        const int16x8x2_t texels =
                        {
                            {
                                vshlq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(texels_u8))), b),
                                vshlq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(texels_u8))), b)
                            }
                        };

                        vst1q_s16(reinterpret_cast<int16_t *>(output.ptr()), texels.val[0]);
                        vst1q_s16(reinterpret_cast<int16_t *>(output.ptr()) + 8, texels.val[1]);
                    },
                    input, output);
                    break;
                }
                case DataType::S32:
                {
                    /* Up-conversion U8 -> S32 */
                    execute_window_loop(window, [&](const Coordinates & id)
                    {
                        const uint8x16_t texels_u8 = vld1q_u8(input.ptr());

                        const int16x8x2_t texels =
                        {
                            {
                                vshlq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(texels_u8))), b),
                                vshlq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(texels_u8))), b)
                            }
                        };

                        vst1q_s32(reinterpret_cast<int32_t *>(output.ptr()), vmovl_s16(vget_low_s16(texels.val[0])));
                        vst1q_s32(reinterpret_cast<int32_t *>(output.ptr()) + 4, vmovl_s16(vget_high_s16(texels.val[0])));
                        vst1q_s32(reinterpret_cast<int32_t *>(output.ptr()) + 8, vmovl_s16(vget_low_s16(texels.val[1])));
                        vst1q_s32(reinterpret_cast<int32_t *>(output.ptr()) + 12, vmovl_s16(vget_high_s16(texels.val[1])));
                    },
                    input, output);
                    break;
                }
                case DataType::U16:
                {
                    /* Up-conversion U8 -> U16 */
                    execute_window_loop(window, [&](const Coordinates & id)
                    {
                        const uint8x16_t texels_u8 = vld1q_u8(input.ptr());

                        const uint16x8x2_t texels =
                        {
                            {
                                vshlq_u16(vmovl_u8(vget_low_u8(texels_u8)), b),
                                vshlq_u16(vmovl_u8(vget_high_u8(texels_u8)), b)
                            }
                        };

                        vst1q_u16(reinterpret_cast<uint16_t *>(output.ptr()), texels.val[0]);
                        vst1q_u16(reinterpret_cast<uint16_t *>(output.ptr()) + 8, texels.val[1]);
                    },
                    input, output);
                    break;
                }
                default:
                    ARM_COMPUTE_ERROR("Output data type not supported");
            }
            break;
        }
        case DataType::QS8:
        {
            switch(_output->info()->data_type())
            {
                case DataType::QS8:
                {
                    const int relative_shift = _fixed_point_position_output - _fixed_point_position_input;
                    /* Fixed point position conversion QS8 -> QS8 */
                    if(relative_shift != 0 || !in_place)
                    {
                        const auto relative_shift_vec = vdupq_n_qs8(relative_shift);
                        execute_window_loop(window, [&](const Coordinates & id)
                        {
                            const qint8x16_t texels_qs8 = vld1q_qs8(reinterpret_cast<const qint8_t *>(input.ptr()));
                            vst1q_qs8(reinterpret_cast<qint8_t *>(output.ptr()), vqrshlq_s8(texels_qs8, relative_shift_vec));
                        },
                        input, output);
                    }
                    break;
                }
                case DataType::F32:
                {
                    /* Up-conversion QS8 -> F32 */
                    execute_window_loop(window, [&](const Coordinates & id)
                    {
                        const qint8x16_t texels_qs8 = vld1q_qs8(reinterpret_cast<const qint8_t *>(input.ptr()));

                        float32x4x2_t texels_low  = vcvt_f32_qs8(vget_low_s8(texels_qs8), _fixed_point_position_input);
                        float32x4x2_t texels_high = vcvt_f32_qs8(vget_high_s8(texels_qs8), _fixed_point_position_input);

                        vst1q_f32(reinterpret_cast<float *>(output.ptr()), texels_low.val[0]);
                        vst1q_f32(reinterpret_cast<float *>(output.ptr()) + 4, texels_low.val[1]);
                        vst1q_f32(reinterpret_cast<float *>(output.ptr()) + 8, texels_high.val[0]);
                        vst1q_f32(reinterpret_cast<float *>(output.ptr()) + 12, texels_high.val[1]);
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
                case DataType::U8:
                {
                    const int16x8_t b = vdupq_n_s16(-static_cast<int16_t>(_shift));

                    /* Down-conversion S16 -> U8 */
                    if(ConvertPolicy::SATURATE == _policy)
                    {
                        execute_window_loop(window, [&](const Coordinates & id)
                        {
                            const int16x8x2_t texels =
                            {
                                {
                                    vqshlq_s16(vld1q_s16(reinterpret_cast<int16_t *>(input.ptr())), b),
                                    vqshlq_s16(vld1q_s16(reinterpret_cast<int16_t *>(input.ptr()) + 8), b)
                                }
                            };

                            vst1q_u8(output.ptr(), vcombine_u8(vqmovun_s16(texels.val[0]), vqmovun_s16(texels.val[1])));
                        },
                        input, output);
                    }
                    else
                    {
                        execute_window_loop(window, [&](const Coordinates & id)
                        {
                            const int16x8x2_t texels =
                            {
                                {
                                    vshlq_s16(vld1q_s16(reinterpret_cast<int16_t *>(input.ptr())), b),
                                    vshlq_s16(vld1q_s16(reinterpret_cast<int16_t *>(input.ptr()) + 8), b)
                                }
                            };

                            vst1q_u8(output.ptr(), vcombine_u8(vmovn_u16(vreinterpretq_u16_s16(texels.val[0])),
                                                               vmovn_u16(vreinterpretq_u16_s16(texels.val[1]))));
                        },
                        input, output);
                    }
                    break;
                }
                case DataType::S32:
                {
                    const int32x4_t b = vdupq_n_s32(_shift);

                    /* Up-conversion S16 -> S32 */
                    execute_window_loop(window, [&](const Coordinates & id)
                    {
                        const int16x8x2_t texels =
                        {
                            {
                                vld1q_s16(reinterpret_cast<int16_t *>(input.ptr())),
                                vld1q_s16(reinterpret_cast<int16_t *>(input.ptr()) + 8)
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

                        vst1q_s32(reinterpret_cast<int32_t *>(output.ptr()), texels_s32.val[0]);
                        vst1q_s32(reinterpret_cast<int32_t *>(output.ptr()) + 4, texels_s32.val[1]);
                        vst1q_s32(reinterpret_cast<int32_t *>(output.ptr()) + 8, texels_s32.val[2]);
                        vst1q_s32(reinterpret_cast<int32_t *>(output.ptr()) + 12, texels_s32.val[3]);
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
                        execute_window_loop(window, [&](const Coordinates & id)
                        {
                            const uint16x8x2_t texels =
                            {
                                {
                                    vqshlq_u16(vld1q_u16(reinterpret_cast<uint16_t *>(input.ptr())), b),
                                    vqshlq_u16(vld1q_u16(reinterpret_cast<uint16_t *>(input.ptr()) + 8), b)
                                }
                            };

                            vst1q_u8(output.ptr(), vcombine_u8(vqmovn_u16(texels.val[0]), vqmovn_u16(texels.val[1])));
                        },
                        input, output);
                    }
                    else
                    {
                        execute_window_loop(window, [&](const Coordinates & id)
                        {
                            const uint16x8x2_t texels =
                            {
                                {
                                    vshlq_u16(vld1q_u16(reinterpret_cast<uint16_t *>(input.ptr())), b),
                                    vshlq_u16(vld1q_u16(reinterpret_cast<uint16_t *>(input.ptr()) + 8), b)
                                }
                            };

                            vst1q_u8(output.ptr(), vcombine_u8(vmovn_u16(texels.val[0]), vmovn_u16(texels.val[1])));
                        },
                        input, output);
                    }
                    break;
                }
                case DataType::U32:
                {
                    const int32x4_t b = vdupq_n_s32(_shift);

                    /* Up-conversion U16 -> U32 */
                    execute_window_loop(window, [&](const Coordinates & id)
                    {
                        const uint16x8x2_t texels =
                        {
                            {
                                vld1q_u16(reinterpret_cast<uint16_t *>(input.ptr())),
                                vld1q_u16(reinterpret_cast<uint16_t *>(input.ptr()) + 8)
                            }
                        };

                        vst1q_u32(reinterpret_cast<uint32_t *>(output.ptr()), vshlq_u32(vmovl_u16(vget_low_u16(texels.val[0])), b));
                        vst1q_u32(reinterpret_cast<uint32_t *>(output.ptr()) + 4, vshlq_u32(vmovl_u16(vget_high_u16(texels.val[0])), b));
                        vst1q_u32(reinterpret_cast<uint32_t *>(output.ptr()) + 8, vshlq_u32(vmovl_u16(vget_low_u16(texels.val[1])), b));
                        vst1q_u32(reinterpret_cast<uint32_t *>(output.ptr()) + 12, vshlq_u32(vmovl_u16(vget_high_u16(texels.val[1])), b));
                    },
                    input, output);
                    break;
                }
                default:
                    ARM_COMPUTE_ERROR("Output data type not supported");
            }
            break;
        }
        case DataType::QS16:
        {
            switch(_output->info()->data_type())
            {
                case DataType::QS16:
                {
                    const int relative_shift = _fixed_point_position_output - _fixed_point_position_input;
                    /* Fixed point position conversion QS16 -> QS16 */
                    if(relative_shift != 0 || !in_place)
                    {
                        const auto relative_shift_vec = vdupq_n_qs16(relative_shift);
                        execute_window_loop(window, [&](const Coordinates & id)
                        {
                            const qint16x8x2_t texels_qs16 =
                            {
                                {
                                    vld1q_qs16(reinterpret_cast<qint16_t *>(input.ptr())),
                                    vld1q_qs16(reinterpret_cast<qint16_t *>(input.ptr()) + 8)
                                }
                            };
                            vst1q_qs16(reinterpret_cast<qint16_t *>(output.ptr()), vqrshlq_s16(texels_qs16.val[0], relative_shift_vec));
                            vst1q_qs16(reinterpret_cast<qint16_t *>(output.ptr()) + 8, vqrshlq_s16(texels_qs16.val[1], relative_shift_vec));
                        },
                        input, output);
                    }
                    break;
                }
                case DataType::F32:
                {
                    /* Up-conversion QS16 -> F32 */
                    execute_window_loop(window, [&](const Coordinates & id)
                    {
                        const int16x8x2_t texels_qs16 =
                        {
                            {
                                vld1q_s16(reinterpret_cast<qint16_t *>(input.ptr())),
                                vld1q_s16(reinterpret_cast<qint16_t *>(input.ptr()) + 8)
                            }
                        };

                        vst1q_f32(reinterpret_cast<float *>(output.ptr()), vcvt_f32_qs16(vget_low_s16(texels_qs16.val[0]), _fixed_point_position_input));
                        vst1q_f32(reinterpret_cast<float *>(output.ptr()) + 4, vcvt_f32_qs16(vget_high_s16(texels_qs16.val[0]), _fixed_point_position_input));
                        vst1q_f32(reinterpret_cast<float *>(output.ptr()) + 8, vcvt_f32_qs16(vget_low_s16(texels_qs16.val[1]), _fixed_point_position_input));
                        vst1q_f32(reinterpret_cast<float *>(output.ptr()) + 12, vcvt_f32_qs16(vget_high_s16(texels_qs16.val[1]), _fixed_point_position_input));
                    },
                    input, output);
                    break;
                }
                default:
                    ARM_COMPUTE_ERROR("Output data type not supported");
            }
            break;
        }
        case DataType::F32:
        {
            switch(_output->info()->data_type())
            {
                case DataType::QS8:
                {
                    /* Down-conversion F32 -> QS8 */
                    execute_window_loop(window, [&](const Coordinates & id)
                    {
                        const float32x4x4_t texels_f32 =
                        {
                            {
                                vld1q_f32(reinterpret_cast<const float *>(input.ptr())),
                                vld1q_f32(reinterpret_cast<const float *>(input.ptr()) + 4),
                                vld1q_f32(reinterpret_cast<const float *>(input.ptr()) + 8),
                                vld1q_f32(reinterpret_cast<const float *>(input.ptr()) + 12)
                            }
                        };

                        const qint8x16_t texels_s8 = vqcvtq_qs8_f32(texels_f32, _fixed_point_position_output);

                        vst1q_s8(reinterpret_cast<int8_t *>(output.ptr()), texels_s8);
                    },
                    input, output);
                    break;
                }
                case DataType::QS16:
                {
                    /* Down-conversion F32 -> QS16 */
                    execute_window_loop(window, [&](const Coordinates & id)
                    {
                        const float32x4x2_t texels_f32_1 =
                        {
                            {
                                vld1q_f32(reinterpret_cast<const float *>(input.ptr())),
                                vld1q_f32(reinterpret_cast<const float *>(input.ptr()) + 4),
                            }
                        };
                        const float32x4x2_t texels_f32_2 =
                        {
                            {
                                vld1q_f32(reinterpret_cast<const float *>(input.ptr()) + 8),
                                vld1q_f32(reinterpret_cast<const float *>(input.ptr()) + 12)
                            }
                        };

                        vst1q_s16(reinterpret_cast<qint16_t *>(output.ptr()), vqcvtq_qs16_f32(texels_f32_1, _fixed_point_position_output));
                        vst1q_s16(reinterpret_cast<qint16_t *>(output.ptr()) + 8, vqcvtq_qs16_f32(texels_f32_2, _fixed_point_position_output));
                    },
                    input, output);
                    break;
                }
                default:
                    ARM_COMPUTE_ERROR("Output data type not supported");
            }
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not supported");
    }
}
