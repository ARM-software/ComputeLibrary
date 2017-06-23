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
#include "arm_compute/core/NEON/kernels/NEDepthConvertKernel.h"

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

NEDepthConvertKernel::NEDepthConvertKernel()
    : _policy(), _shift(0)
{
}

void NEDepthConvertKernel::configure(const ITensor *input, ITensor *output, ConvertPolicy policy, uint32_t shift)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::QS8, DataType::S16, DataType::U16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8, DataType::QS8, DataType::S16, DataType::U16, DataType::U32, DataType::S32, DataType::F32);
    ARM_COMPUTE_ERROR_ON(shift >= 8);
    ARM_COMPUTE_ERROR_ON(input == output);
    ARM_COMPUTE_ERROR_ON_MSG(input->info()->data_type() == output->info()->data_type(), "Input and output data_types must be different");

    ARM_COMPUTE_ERROR_ON_MSG(input->info()->data_type() == DataType::QS8 && (output->info()->data_type() != DataType::F32),
                             "Only data_types supported [in] QS8 ->  [out] F32");

    ARM_COMPUTE_ERROR_ON_MSG(input->info()->data_type() == DataType::U8 && (output->info()->data_type() != DataType::S16 && output->info()->data_type() != DataType::U16
                                                                            && output->info()->data_type() != DataType::S32),
                             "Only data_types supported [in] U8 -> [out] U16, S16, S32");

    ARM_COMPUTE_ERROR_ON_MSG(input->info()->data_type() == DataType::U16 && (output->info()->data_type() != DataType::U8 && output->info()->data_type() != DataType::U32),
                             "Only data_types supported [in] U16 ->  [out] U8, U32");

    ARM_COMPUTE_ERROR_ON_MSG(input->info()->data_type() == DataType::S16 && (output->info()->data_type() != DataType::U8 && output->info()->data_type() != DataType::S32),
                             "Only data_types supported [in] S16 ->  [out] U8, S32");

    ARM_COMPUTE_ERROR_ON_MSG(input->info()->data_type() == DataType::F32 && (output->info()->data_type() != DataType::QS8),
                             "Only data_types supported [in] F32 ->  [out] QS8");

    _policy = policy;
    _shift  = shift;

    constexpr unsigned int num_elems_processed_per_iteration = 16;
    INESimpleKernel::configure(input, output, num_elems_processed_per_iteration);
}

void NEDepthConvertKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INESimpleKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(nullptr == _input);
    ARM_COMPUTE_ERROR_ON(nullptr == _output);
    ARM_COMPUTE_ERROR_ON(_input == _output);

    Iterator input(_input, window);
    Iterator output(_output, window);

    switch(_input->info()->data_type())
    {
        case DataType::QS8:
        {
            const int fixed_point_position = _input->info()->fixed_point_position();

            switch(_output->info()->data_type())
            {
                case DataType::F32:
                {
                    /* Up-conversion QS8 -> F32 */
                    execute_window_loop(window, [&](const Coordinates & id)
                    {
                        const int8x16_t texels_s8 = vld1q_s8(reinterpret_cast<const int8_t *>(input.ptr()));

                        float32x4x2_t texels_low  = vcvt_f32_qs8(vget_low_s8(texels_s8), fixed_point_position);
                        float32x4x2_t texels_high = vcvt_f32_qs8(vget_high_s8(texels_s8), fixed_point_position);

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
        case DataType::F32:
        {
            switch(_output->info()->data_type())
            {
                case DataType::QS8:
                {
                    const int fixed_point_position = _output->info()->fixed_point_position();
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

                        const qint8x16_t texels_s8 = vcvtq_qs8_f32(texels_f32, fixed_point_position);

                        vst1q_s8(reinterpret_cast<int8_t *>(output.ptr()), texels_s8);
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
