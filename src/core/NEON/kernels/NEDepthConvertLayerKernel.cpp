/*
 * Copyright (c) 2016-2018 ARM Limited.
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

#include <arm_neon.h>

using namespace arm_compute;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, ConvertPolicy policy, uint32_t shift)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(output);
    ARM_COMPUTE_UNUSED(policy);
    ARM_COMPUTE_RETURN_ERROR_ON(input == output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::U8, DataType::S16, DataType::U16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::QASYMM8, DataType::U8, DataType::S16, DataType::U16, DataType::U32, DataType::S32, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(shift >= 8);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::QASYMM8 && (output->data_type() != DataType::F16 && output->data_type() != DataType::F32),
                                    "Only data_types supported [in] QASYMM8 -> [out] F16, F32");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::U8 && (output->data_type() != DataType::S16 && output->data_type() != DataType::U16
                                                                           && output->data_type() != DataType::S32),
                                    "Only data_types supported [in] U8 -> [out] U16, S16, S32");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::U16 && (output->data_type() != DataType::U8 && output->data_type() != DataType::U32),
                                    "Only data_types supported [in] U16 ->  [out] U8, U32");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::S16 && (output->data_type() != DataType::U8 && output->data_type() != DataType::S32),
                                    "Only data_types supported [in] S16 ->  [out] U8, S32");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::F16 && (output->data_type() != DataType::QASYMM8 && output->data_type() != DataType::F32),
                                    "Only data_types supported [in] F16 ->  [out] QASYMM8, F32");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::F32 && (output->data_type() != DataType::QASYMM8 && output->data_type() != DataType::F16),
                                    "Only data_types supported [in] F32 ->  [out] QASYMM8, F16");

    // Validate in case of configured output
    if(output->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    constexpr unsigned int num_elems_processed_per_iteration = 16;

    Window win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
    bool                   window_changed = update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, output->valid_region());

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
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
    auto win_config = validate_and_configure_window(input->info(), output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICPPKernel::configure(win_config.second);
}

Status NEDepthConvertLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, ConvertPolicy policy, uint32_t shift)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, policy, shift));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get()).first);

    return Status{};
}

void NEDepthConvertLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);
    ARM_COMPUTE_ERROR_ON_NULLPTR(_input, _output);
    ARM_COMPUTE_ERROR_ON(_input == _output);

    Iterator input(_input, window);
    Iterator output(_output, window);

    switch(_input->info()->data_type())
    {
        case DataType::QASYMM8:
        {
            switch(_output->info()->data_type())
            {
                /* Up-conversion QASYMM8 -> F32 */
                case DataType::F32:
                {
                    const float32x4_t scale  = vdupq_n_f32(_input->info()->quantization_info().scale);
                    const int32x4_t   offset = vdupq_n_s32(_input->info()->quantization_info().offset);

                    execute_window_loop(window, [&](const Coordinates & id)
                    {
                        const uint8x16_t   texels_u8 = vld1q_u8(input.ptr());
                        const uint16x8x2_t texels_u16 =
                        {
                            {
                                vmovl_u8(vget_low_u8(texels_u8)),
                                vmovl_u8(vget_high_u8(texels_u8))
                            }
                        };

                        const int32x4x4_t texels_s32 =
                        {
                            {
                                vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(texels_u16.val[0]))),
                                vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(texels_u16.val[0]))),
                                vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(texels_u16.val[1]))),
                                vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(texels_u16.val[1])))
                            }
                        };

                        vst1q_f32(reinterpret_cast<float *>(output.ptr()), vmulq_f32(vcvtq_f32_s32(vsubq_s32(texels_s32.val[0], offset)), scale));
                        vst1q_f32(reinterpret_cast<float *>(output.ptr()) + 4, vmulq_f32(vcvtq_f32_s32(vsubq_s32(texels_s32.val[1], offset)), scale));
                        vst1q_f32(reinterpret_cast<float *>(output.ptr()) + 8, vmulq_f32(vcvtq_f32_s32(vsubq_s32(texels_s32.val[2], offset)), scale));
                        vst1q_f32(reinterpret_cast<float *>(output.ptr()) + 12, vmulq_f32(vcvtq_f32_s32(vsubq_s32(texels_s32.val[3], offset)), scale));
                    },
                    input, output);
                    break;
                }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                /* Up-conversion QASYMM8 -> F16 */
                case DataType::F16:
                {
                    const float16x8_t scale  = vdupq_n_f16(static_cast<float16_t>(_input->info()->quantization_info().scale));
                    const int16x8_t   offset = vdupq_n_s16(static_cast<int16_t>(_input->info()->quantization_info().offset));

                    execute_window_loop(window, [&](const Coordinates & id)
                    {
                        const uint8x16_t  texels_u8 = vld1q_u8(input.ptr());
                        const int16x8x2_t texels_s16 =
                        {
                            {
                                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(texels_u8))),
                                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(texels_u8)))
                            }
                        };

                        vst1q_f16(reinterpret_cast<float16_t *>(output.ptr()), vmulq_f16(vcvtq_f16_s16(vsubq_s16(texels_s16.val[0], offset)), scale));
                        vst1q_f16(reinterpret_cast<float16_t *>(output.ptr()) + 8, vmulq_f16(vcvtq_f16_s16(vsubq_s16(texels_s16.val[1], offset)), scale));
                    },
                    input, output);
                    break;
                }
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
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
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            switch(_output->info()->data_type())
            {
                case DataType::QASYMM8:
                {
                    const float16x8_t scale        = vinvq_f16(vdupq_n_f16(static_cast<float16_t>(_output->info()->quantization_info().scale)));
                    const int16x8_t   offset       = vdupq_n_s16(static_cast<int16_t>(_output->info()->quantization_info().offset));
                    const int16x8_t   max_val_vec  = vdupq_n_s16(255);
                    const int16x8_t   zero_val_vec = vdupq_n_s16(0);

                    /* Down-conversion F16 -> QASYMM8 */
                    execute_window_loop(window, [&](const Coordinates & id)
                    {
                        const float16x8x2_t texels =
                        {
                            {
                                vmulq_f16(vld1q_f16(reinterpret_cast<float16_t *>(input.ptr())), scale),
                                vmulq_f16(vld1q_f16(reinterpret_cast<float16_t *>(input.ptr()) + 8), scale),
                            }
                        };

                        const auto texel_quantized_0 = vmaxq_s16(vminq_s16(vaddq_s16(vcvtq_s16_f16(texels.val[0]), offset), max_val_vec), zero_val_vec);
                        const auto texel_quantized_1 = vmaxq_s16(vminq_s16(vaddq_s16(vcvtq_s16_f16(texels.val[1]), offset), max_val_vec), zero_val_vec);
                        vst1q_u8(reinterpret_cast<uint8_t *>(output.ptr()), vcombine_u8(vqmovun_s16(texel_quantized_0), vqmovun_s16(texel_quantized_1)));
                    },
                    input, output);
                    break;
                }
                case DataType::F32:
                {
                    const float32x4_t scale = vdupq_n_f32(1 << _shift);

                    /* Up-conversion F16 -> F32 */
                    execute_window_loop(window, [&](const Coordinates & id)
                    {
                        const float16x8x2_t texels =
                        {
                            {
                                vld1q_f16(reinterpret_cast<float16_t *>(input.ptr())),
                                vld1q_f16(reinterpret_cast<float16_t *>(input.ptr()) + 8)
                            }
                        };

                        vst1q_f32(reinterpret_cast<float *>(output.ptr()), vmulq_f32(vcvt_f32_f16(vget_low_f16(texels.val[0])), scale));
                        vst1q_f32(reinterpret_cast<float *>(output.ptr()) + 4, vmulq_f32(vcvt_f32_f16(vget_high_f16(texels.val[0])), scale));
                        vst1q_f32(reinterpret_cast<float *>(output.ptr()) + 8, vmulq_f32(vcvt_f32_f16(vget_low_f16(texels.val[1])), scale));
                        vst1q_f32(reinterpret_cast<float *>(output.ptr()) + 12, vmulq_f32(vcvt_f32_f16(vget_high_f16(texels.val[1])), scale));
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
                case DataType::QASYMM8:
                {
                    const float32x4_t scale        = vinvq_f32(vdupq_n_f32(_output->info()->quantization_info().scale));
                    const int32x4_t   offset       = vdupq_n_s32(_output->info()->quantization_info().offset);
                    const int32x4_t   max_val_vec  = vdupq_n_s32(255);
                    const int32x4_t   zero_val_vec = vdupq_n_s32(0);

                    /* Down-conversion F32 -> QASYMM8 */
                    execute_window_loop(window, [&](const Coordinates & id)
                    {
                        const float32x4x4_t texels =
                        {
                            {
                                vmulq_f32(vld1q_f32(reinterpret_cast<float *>(input.ptr())), scale),
                                vmulq_f32(vld1q_f32(reinterpret_cast<float *>(input.ptr()) + 4), scale),
                                vmulq_f32(vld1q_f32(reinterpret_cast<float *>(input.ptr()) + 8), scale),
                                vmulq_f32(vld1q_f32(reinterpret_cast<float *>(input.ptr()) + 12), scale)
                            }
                        };

                        const auto texel_quantized_0 = vmaxq_s32(vminq_s32(vaddq_s32(vcvtq_s32_f32(texels.val[0]), offset), max_val_vec), zero_val_vec);
                        const auto texel_quantized_1 = vmaxq_s32(vminq_s32(vaddq_s32(vcvtq_s32_f32(texels.val[1]), offset), max_val_vec), zero_val_vec);
                        const auto texel_quantized_2 = vmaxq_s32(vminq_s32(vaddq_s32(vcvtq_s32_f32(texels.val[2]), offset), max_val_vec), zero_val_vec);
                        const auto texel_quantized_3 = vmaxq_s32(vminq_s32(vaddq_s32(vcvtq_s32_f32(texels.val[3]), offset), max_val_vec), zero_val_vec);

                        const auto converted_0 = vqmovn_u16(vcombine_u16(vqmovun_s32(texel_quantized_0), vqmovun_s32(texel_quantized_1)));
                        const auto converted_1 = vqmovn_u16(vcombine_u16(vqmovun_s32(texel_quantized_2), vqmovun_s32(texel_quantized_3)));

                        vst1q_u8(reinterpret_cast<uint8_t *>(output.ptr()), vcombine_u8(converted_0, converted_1));
                    },
                    input, output);
                    break;
                }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                case DataType::F16:
                {
                    const float32x4_t scale = vdupq_n_f32(1.f / (1 << _shift));

                    /* Down-conversion F32 -> F16 */
                    execute_window_loop(window, [&](const Coordinates & id)
                    {
                        const float32x4x4_t texels =
                        {
                            {
                                vmulq_f32(vld1q_f32(reinterpret_cast<float *>(input.ptr())), scale),
                                vmulq_f32(vld1q_f32(reinterpret_cast<float *>(input.ptr()) + 4), scale),
                                vmulq_f32(vld1q_f32(reinterpret_cast<float *>(input.ptr()) + 8), scale),
                                vmulq_f32(vld1q_f32(reinterpret_cast<float *>(input.ptr()) + 12), scale)
                            }
                        };

                        vst1q_f16(reinterpret_cast<float16_t *>(output.ptr()), vcombine_f16(vcvt_f16_f32(texels.val[0]), vcvt_f16_f32(texels.val[1])));
                        vst1q_f16(reinterpret_cast<float16_t *>(output.ptr()) + 8, vcombine_f16(vcvt_f16_f32(texels.val[2]), vcvt_f16_f32(texels.val[3])));
                    },
                    input, output);
                    break;
                }
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
                default:
                    ARM_COMPUTE_ERROR("Output data type not supported");
            }
            break;
        default:
            ARM_COMPUTE_ERROR("Not supported");
    }
}
