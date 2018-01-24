/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEDirectConvolutionLayerOutputStageKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEAsymm.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

using namespace arm_compute;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QASYMM8,
                                                         DataType::QS16, DataType::F16,
                                                         DataType::QS32, DataType::S32, DataType::F32);

    if(bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(bias, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::QS32, DataType::S32, DataType::F32);

        if(is_data_type_fixed_point(input->data_type()))
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::QS8 && bias->data_type() != DataType::QS8, "Wrong data type for bias");
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::QS16 && bias->data_type() != DataType::QS8, "Wrong data type for bias");
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::QS32 && bias->data_type() != DataType::QS16, "Wrong data type for bias");
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT_POSITION(input, bias);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, bias);
        }

        ARM_COMPUTE_RETURN_ERROR_ON(bias->num_dimensions() > 1);
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(!is_data_type_quantized(input->data_type()), "Calling output stage kernel with floating point arguments");
    }

    // Checks performed when output is configured
    if((output != nullptr) && (output->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::QS8, DataType::QASYMM8, DataType::QS16, DataType::F32);
        if(is_data_type_fixed_point(input->data_type()))
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::QS8 && output->data_type() != DataType::QS8, "Wrong data type for output");
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::QS16 && output->data_type() != DataType::QS8, "Wrong data type for output");
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::QS32 && output->data_type() != DataType::QS16, "Wrong data type for output");
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT_POSITION(input, output);
        }
        else if(is_data_type_quantized_asymmetric(output->data_type()))
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::S32 && output->data_type() != DataType::QASYMM8, "Wrong data type for bias");
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        }
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *bias, ITensorInfo *output)
{
    bool         window_changed                    = false;
    unsigned int num_elems_processed_per_iteration = 16 / element_size_from_data_type(input->data_type());

    // Update processed elements when input is S32 (comes from quantization input)
    if(input->data_type() == DataType::S32)
    {
        num_elems_processed_per_iteration = 16;
    }

    // Configure kernel window
    Window                 win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);

    if(output != nullptr && (output->total_size() != 0))
    {
        AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);

        if(bias == nullptr)
        {
            window_changed = update_window_and_padding(win, input_access, output_access);
        }
        else
        {
            AccessWindowStatic bias_access(bias, 0, 0, bias->dimension(0), bias->dimension(1));
            window_changed = update_window_and_padding(win, input_access, output_access, bias_access);
        }

        output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));
    }
    else
    {
        if(bias == nullptr)
        {
            window_changed = update_window_and_padding(win, input_access);
        }
        else
        {
            AccessWindowStatic bias_access(bias, 0, 0, bias->dimension(0), bias->dimension(1));
            window_changed = update_window_and_padding(win, input_access, bias_access);
        }

        input_access.set_valid_region(win, ValidRegion(Coordinates(), input->tensor_shape()));
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

// Internal load
inline float32x4_t internal_vld1q(const float *in)
{
    return vld1q_f32(in);
}
inline qint8x16_t internal_vld1q(const qint8_t *in)
{
    return vld1q_qs8(in);
}
inline qint16x8_t internal_vld1q(const qint16_t *in)
{
    return vld1q_qs16(in);
}
inline qint32x4_t internal_vld1q(const qint32_t *in)
{
    return vld1q_s32(in);
}

// Internal store
inline void internal_vst1q(float *p, const float32x4_t &v)
{
    vst1q_f32(p, v);
}
inline void internal_vst1q(qint8_t *p, const qint8x16_t &v)
{
    vst1q_qs8(p, v);
}
inline void internal_vst1q(qint8_t *p, const qint16x8_t &v)
{
    vst1_qs8(p, vqmovn_s16(v));
}
inline void internal_vst1q(qint16_t *p, const qint16x8_t &v)
{
    vst1q_qs16(p, v);
}
inline void internal_vst1q(qint32_t *p, const qint32x4_t &v)
{
    vst1q_s32(p, v);
}

inline void internal_vst1q(qint16_t *p, const qint32x4_t &v)
{
    vst1_qs16(p, vqmovn_qs32(v));
}

// Internal vdup
inline float32x4_t internal_vdupq_n(float v)
{
    return vdupq_n_f32(v);
}
inline qint8x16_t internal_vdupq_n(qint8_t v)
{
    return vdupq_n_qs8(v);
}
inline qint16x8_t internal_vdupq_n(qint16_t v)
{
    return vdupq_n_qs16(v);
}
inline qint32x4_t internal_vdupq_n(qint32_t v)
{
    return vdupq_n_qs32(v);
}

// Internal vadd
inline float32x4_t internal_vqaddq(const float32x4_t &x, const float32x4_t &y)
{
    return vaddq_f32(x, y);
}
inline qint8x16_t internal_vqaddq(const qint8x16_t &x, const qint8x16_t &y)
{
    return vqaddq_qs8(x, y);
}
inline qint16x8_t internal_vqaddq(const qint16x8_t &x, const qint16x8_t &y)
{
    return vqaddq_qs16(x, y);
}
inline qint32x4_t internal_vqaddq(const qint32x4_t &x, const qint32x4_t &y)
{
    return vqaddq_qs32(x, y);
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
inline float16x8_t internal_vld1q(const float16_t *in)
{
    return vld1q_f16(in);
}
inline void internal_vst1q(float16_t *p, const float16x8_t &v)
{
    vst1q_f16(p, v);
}
inline float16x8_t internal_vdupq_n(float16_t v)
{
    return vdupq_n_f16(v);
}
inline float16x8_t internal_vqaddq(const float16x8_t &x, const float16x8_t &y)
{
    return vaddq_f16(x, y);
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

template <typename T1, typename T2, bool in_place, bool has_bias>
void output_stage(ITensor *input, const ITensor *bias, const Window &window, ITensor *output,
                  int result_fixedpoint_multiplier, int result_shift, int result_offset_after_shift)
{
    ARM_COMPUTE_UNUSED(result_fixedpoint_multiplier);
    ARM_COMPUTE_UNUSED(result_shift);
    ARM_COMPUTE_UNUSED(result_offset_after_shift);

    Iterator in(input, window);

    if(in_place) // In place accumulate
    {
        execute_window_loop(window, [&](const Coordinates & id)
        {
            // Get bias and pointer to input
            const auto in_ptr = reinterpret_cast<T1 *>(in.ptr());

            // Accumulate bias
            if(has_bias)
            {
                const auto vb = internal_vdupq_n(static_cast<T1>(*reinterpret_cast<const T2 *>(bias->ptr_to_element(Coordinates(id.z())))));
                internal_vst1q(in_ptr, internal_vqaddq(internal_vld1q(in_ptr), vb));
            }
            else
            {
                internal_vst1q(in_ptr, internal_vld1q(in_ptr));
            }
        },
        in);
    }
    else // Out of place accumulate
    {
        Iterator out(output, window);
        execute_window_loop(window, [&](const Coordinates & id)
        {
            // Get bias and pointer to input
            const auto in_ptr  = reinterpret_cast<const T1 *>(in.ptr());
            const auto out_ptr = reinterpret_cast<T2 *>(out.ptr());

            // Accumulate bias
            if(has_bias)
            {
                const auto vb = internal_vdupq_n(static_cast<T1>(*reinterpret_cast<const T2 *>(bias->ptr_to_element(Coordinates(id.z())))));
                internal_vst1q(out_ptr, internal_vqaddq(internal_vld1q(in_ptr), vb));
            }
            else
            {
                internal_vst1q(out_ptr, internal_vld1q(in_ptr));
            }
        },
        in, out);
    }
}

// QASYMM8 specializations
template <>
void output_stage<int32_t, uint8_t, false, true>(ITensor *input, const ITensor *bias, const Window &window, ITensor *output,
                                                 int result_fixedpoint_multiplier, int result_shift, int result_offset_after_shift)
{
    const int32x4_t result_offset_after_shift_s32 = vdupq_n_s32(result_offset_after_shift);
    uint8x16_t      min                           = vdupq_n_u8(0);
    uint8x16_t      max                           = vdupq_n_u8(255);

    Iterator in(input, window);
    Iterator out(output, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        // Get bias and pointer to input
        const auto  in_ptr = reinterpret_cast<int32_t *>(in.ptr());
        int32x4x4_t v_in =
        {
            {
                vld1q_s32(in_ptr),
                vld1q_s32(in_ptr + 4),
                vld1q_s32(in_ptr + 8),
                vld1q_s32(in_ptr + 12)
            }
        };

        // Accumulate bias
        const auto vb = vdupq_n_s32(*reinterpret_cast<const int32_t *>(bias->ptr_to_element(Coordinates(id.z()))));
        v_in =
        {
            {
                vaddq_s32(v_in.val[0], vb),
                vaddq_s32(v_in.val[1], vb),
                vaddq_s32(v_in.val[2], vb),
                vaddq_s32(v_in.val[3], vb)
            }
        };

        const auto out_ptr = reinterpret_cast<uint8_t *>(out.ptr());
        vst1q_u8(out_ptr, finalize_quantization<false>(v_in, result_fixedpoint_multiplier, result_shift, result_offset_after_shift_s32, min, max));
    },
    in, out);
}
template <>
void output_stage<int32_t, uint8_t, false, false>(ITensor *input, const ITensor *bias, const Window &window, ITensor *output,
                                                  int result_fixedpoint_multiplier, int result_shift, int result_offset_after_shift)
{
    ARM_COMPUTE_UNUSED(bias);

    const int32x4_t result_offset_after_shift_s32 = vdupq_n_s32(result_offset_after_shift);
    uint8x16_t      min                           = vdupq_n_u8(0);
    uint8x16_t      max                           = vdupq_n_u8(255);

    Iterator in(input, window);
    Iterator out(output, window);
    execute_window_loop(window, [&](const Coordinates & id)
    {
        // Get bias and pointer to input
        const auto  in_ptr = reinterpret_cast<int32_t *>(in.ptr());
        int32x4x4_t v_in =
        {
            {
                vld1q_s32(in_ptr),
                vld1q_s32(in_ptr + 4),
                vld1q_s32(in_ptr + 8),
                vld1q_s32(in_ptr + 12)
            }
        };

        const auto out_ptr = reinterpret_cast<uint8_t *>(out.ptr());
        vst1q_u8(out_ptr, finalize_quantization<false>(v_in, result_fixedpoint_multiplier, result_shift, result_offset_after_shift_s32, min, max));
    },
    in, out);
}
} // namespace

NEDirectConvolutionLayerOutputStageKernel::NEDirectConvolutionLayerOutputStageKernel()
    : _func(nullptr), _input(nullptr), _bias(nullptr), _output(nullptr), _result_fixedpoint_multiplier(0), _result_shift(0), _result_offset_after_shift(0)
{
}

void NEDirectConvolutionLayerOutputStageKernel::configure(ITensor *input, const ITensor *bias, ITensor *output,
                                                          int result_fixedpoint_multiplier, int result_shift, int result_offset_after_shift)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input);

    // Auto-initialize output output if required
    if(output != nullptr)
    {
        // Work out expected output data type
        const DataType output_dt = (input->info()->data_type() == DataType::S32) ? DataType::QASYMM8 : input->info()->data_type();
        // Output tensor auto initialization if not yet initialized
        auto_init_if_empty(*output->info(), input->info()->clone()->set_data_type(output_dt));
    }

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), (bias == nullptr) ? nullptr : bias->info(), (output == nullptr) ? nullptr : output->info()));

    _func                         = nullptr;
    _bias                         = bias;
    _input                        = input;
    _output                       = output;
    _result_fixedpoint_multiplier = result_fixedpoint_multiplier;
    _result_shift                 = result_shift;
    _result_offset_after_shift    = result_offset_after_shift;

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), (bias == nullptr) ? nullptr : bias->info(), (output == nullptr) ? nullptr : output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);

    // Set appropriate function
    switch(input->info()->data_type())
    {
        case DataType::QS8:
        {
            if(bias == nullptr)
            {
                _func = (output == nullptr) ? &output_stage<qint8_t, qint8_t, true, false> : &output_stage<qint8_t, qint8_t, false, false>;
            }
            else
            {
                _func = (output == nullptr) ? &output_stage<qint8_t, qint8_t, true, true> : &output_stage<qint8_t, qint8_t, false, true>;
            }
            break;
        }
        case DataType::QS16:
        {
            if(bias != nullptr && bias->info()->data_type() == DataType::QS8)
            {
                _func = (output == nullptr) ? &output_stage<qint16_t, qint8_t, true, true> : &output_stage<qint16_t, qint8_t, false, true>;
            }
            else if(bias == nullptr)
            {
                _func = (output == nullptr) ? &output_stage<qint16_t, qint8_t, true, false> : &output_stage<qint16_t, qint8_t, false, false>;
            }
            else
            {
                ARM_COMPUTE_ERROR("Not implemented");
            }
            break;
        }
        case DataType::QS32:
        {
            _func = (output == nullptr) ? &output_stage<qint32_t, qint16_t, true, true> : &output_stage<qint32_t, qint16_t, false, true>;
            break;
        }
        case DataType::S32:
            _func = (bias == nullptr) ? &output_stage<int32_t, uint8_t, false, false> : &output_stage<int32_t, uint8_t, false, true>;
            break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
        {
            _func = (output == nullptr) ? &output_stage<float16_t, float16_t, true, true> : &output_stage<float16_t, float16_t, false, true>;
            break;
        }
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        case DataType::F32:
        {
            _func = (output == nullptr) ? &output_stage<float, float, true, true> : &output_stage<float, float, false, true>;
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Unsupported combination of types among the inputs.");
        }
    }
}

Status NEDirectConvolutionLayerOutputStageKernel::validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, bias, output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), bias->clone().get(), output == nullptr ? nullptr : output->clone().get()).first);

    return Status{};
}

void NEDirectConvolutionLayerOutputStageKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (*_func)(_input, _bias, window, _output, _result_fixedpoint_multiplier, _result_shift, _result_offset_after_shift);
}
