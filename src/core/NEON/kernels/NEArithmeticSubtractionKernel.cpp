/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEArithmeticSubtractionKernel.h"

#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEAsymm.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"

#include <algorithm>
#include <arm_neon.h>
#include <cstdint>
#include <map>
#include <string>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

namespace
{
constexpr unsigned int num_elems_processed_per_iteration = 16;

void sub_wrap_U8_U8_U8(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    Iterator input1(in1, window.broadcast_if_dimension_le_one(in1->info()->tensor_shape()));
    Iterator input2(in2, window.broadcast_if_dimension_le_one(in2->info()->tensor_shape()));
    Iterator output(out, window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        const uint8x16_t ta1 = vld1q_u8(input1.ptr());
        const uint8x16_t ta2 = vld1q_u8(input2.ptr());

        vst1q_u8(output.ptr(), vsubq_u8(ta1, ta2));
    },
    input1, input2, output);
}

void sub_saturate_U8_U8_U8(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    Iterator input1(in1, window.broadcast_if_dimension_le_one(in1->info()->tensor_shape()));
    Iterator input2(in2, window.broadcast_if_dimension_le_one(in2->info()->tensor_shape()));
    Iterator output(out, window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        const uint8x16_t ta1 = vld1q_u8(input1.ptr());
        const uint8x16_t ta2 = vld1q_u8(input2.ptr());

        vst1q_u8(output.ptr(), vqsubq_u8(ta1, ta2));
    },
    input1, input2, output);
}

void sub_saturate_QAYSMM8_QAYSMM8_QAYSMM8(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    Iterator input1(in1, window.broadcast_if_dimension_le_one(in1->info()->tensor_shape()));
    Iterator input2(in2, window.broadcast_if_dimension_le_one(in2->info()->tensor_shape()));
    Iterator output(out, window);

    const UniformQuantizationInfo iq1_info = in1->info()->quantization_info().uniform();
    const UniformQuantizationInfo iq2_info = in2->info()->quantization_info().uniform();
    const UniformQuantizationInfo oq_info  = out->info()->quantization_info().uniform();

    execute_window_loop(window, [&](const Coordinates &)
    {
        const float32x4x4_t ta1 = vdequantize(vld1q_u8(reinterpret_cast<const qasymm8_t *>(input1.ptr())), iq1_info);
        const float32x4x4_t ta2 = vdequantize(vld1q_u8(reinterpret_cast<const qasymm8_t *>(input2.ptr())), iq2_info);

        const float32x4x4_t ta3 =
        {
            {
                vsubq_f32(ta1.val[0], ta2.val[0]),
                vsubq_f32(ta1.val[1], ta2.val[1]),
                vsubq_f32(ta1.val[2], ta2.val[2]),
                vsubq_f32(ta1.val[3], ta2.val[3]),
            }
        };

        const uint8x16_t result = vquantize(ta3, oq_info);

        vst1q_u8(reinterpret_cast<qasymm8_t *>(output.ptr()), result);
    },
    input1, input2, output);
}

void sub_wrap_S16_S16_S16(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    Iterator input1(in1, window.broadcast_if_dimension_le_one(in1->info()->tensor_shape()));
    Iterator input2(in2, window.broadcast_if_dimension_le_one(in2->info()->tensor_shape()));
    Iterator output(out, window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        const int16x8x2_t ta1 = vld2q_s16(reinterpret_cast<const int16_t *>(input1.ptr()));
        const int16x8x2_t ta2 = vld2q_s16(reinterpret_cast<const int16_t *>(input2.ptr()));

        const int16x8x2_t ta3 =
        {
            {
                vsubq_s16(ta1.val[0], ta2.val[0]),
                vsubq_s16(ta1.val[1], ta2.val[1])
            }
        };

        vst2q_s16(reinterpret_cast<int16_t *>(output.ptr()), ta3);
    },
    input1, input2, output);
}

void sub_saturate_S16_S16_S16(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    Iterator input1(in1, window.broadcast_if_dimension_le_one(in1->info()->tensor_shape()));
    Iterator input2(in2, window.broadcast_if_dimension_le_one(in2->info()->tensor_shape()));
    Iterator output(out, window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        const int16x8x2_t ta1 = vld2q_s16(reinterpret_cast<const int16_t *>(input1.ptr()));
        const int16x8x2_t ta2 = vld2q_s16(reinterpret_cast<const int16_t *>(input2.ptr()));

        const int16x8x2_t ta3 =
        {
            {
                vqsubq_s16(ta1.val[0], ta2.val[0]),
                vqsubq_s16(ta1.val[1], ta2.val[1])
            }
        };

        vst2q_s16(reinterpret_cast<int16_t *>(output.ptr()), ta3);
    },
    input1, input2, output);
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
inline float16x8x2_t vsub2q_f16(const float16x8x2_t &a, const float16x8x2_t &b)
{
    const float16x8x2_t res =
    {
        {
            vsubq_f16(a.val[0], b.val[0]),
            vsubq_f16(a.val[1], b.val[1])
        }
    };

    return res;
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

void sub_F16_F16_F16(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    Iterator input1(in1, window.broadcast_if_dimension_le_one(in1->info()->tensor_shape()));
    Iterator input2(in2, window.broadcast_if_dimension_le_one(in2->info()->tensor_shape()));
    Iterator output(out, window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        const float16x8x2_t a = vld2q_f16(reinterpret_cast<const float16_t *>(input1.ptr()));
        const float16x8x2_t b = vld2q_f16(reinterpret_cast<const float16_t *>(input2.ptr()));

        vst2q_f16(reinterpret_cast<float16_t *>(output.ptr()), vsub2q_f16(a, b));
    },
    input1, input2, output);
#else  /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    ARM_COMPUTE_UNUSED(in1);
    ARM_COMPUTE_UNUSED(in2);
    ARM_COMPUTE_UNUSED(out);
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_ERROR("Not supported, recompile the library with arch=arm64-v8.2-a");
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
}

void sub_F32_F32_F32(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    Iterator input1(in1, window.broadcast_if_dimension_le_one(in1->info()->tensor_shape()));
    Iterator input2(in2, window.broadcast_if_dimension_le_one(in2->info()->tensor_shape()));
    Iterator output(out, window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        const float32x4x4_t ta1 = vld4q_f32(reinterpret_cast<const float *>(input1.ptr()));
        const float32x4x4_t ta2 = vld4q_f32(reinterpret_cast<const float *>(input2.ptr()));

        const float32x4x4_t ta3 =
        {
            {
                vsubq_f32(ta1.val[0], ta2.val[0]),
                vsubq_f32(ta1.val[1], ta2.val[1]),
                vsubq_f32(ta1.val[2], ta2.val[2]),
                vsubq_f32(ta1.val[3], ta2.val[3]),
            }
        };

        vst4q_f32(reinterpret_cast<float *>(output.ptr()), ta3);
    },
    input1, input2, output);
}
void sub_wrap_S16_U8_S16(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    Iterator input1(in1, window.broadcast_if_dimension_le_one(in1->info()->tensor_shape()));
    Iterator input2(in2, window.broadcast_if_dimension_le_one(in2->info()->tensor_shape()));
    Iterator output(out, window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        const uint8x16_t bv_0 = vld1q_u8(input2.ptr());
        int16x8_t        a1_0 = vld1q_s16(reinterpret_cast<const int16_t *>(input1.ptr()));
        int16x8_t        a2_0 = vld1q_s16(reinterpret_cast<const int16_t *>(input1.ptr()) + 8);

        a1_0 = vsubq_s16(a1_0, vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(bv_0))));
        a2_0 = vsubq_s16(a2_0, vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(bv_0))));

        vst1q_s16(reinterpret_cast<int16_t *>(output.ptr()), a1_0);
        vst1q_s16(reinterpret_cast<int16_t *>(output.ptr()) + 8, a2_0);
    },
    input1, input2, output);
}

void sub_saturate_S16_U8_S16(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    Iterator input1(in1, window.broadcast_if_dimension_le_one(in1->info()->tensor_shape()));
    Iterator input2(in2, window.broadcast_if_dimension_le_one(in2->info()->tensor_shape()));
    Iterator output(out, window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        const uint8x16_t bv_0 = vld1q_u8(input2.ptr());
        int16x8_t        a1_0 = vld1q_s16(reinterpret_cast<const int16_t *>(input1.ptr()));
        int16x8_t        a2_0 = vld1q_s16(reinterpret_cast<const int16_t *>(input1.ptr()) + 8);

        a1_0 = vqsubq_s16(a1_0, vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(bv_0))));
        a2_0 = vqsubq_s16(a2_0, vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(bv_0))));

        vst1q_s16(reinterpret_cast<int16_t *>(output.ptr()), a1_0);
        vst1q_s16(reinterpret_cast<int16_t *>(output.ptr()) + 8, a2_0);
    },
    input1, input2, output);
}

void sub_wrap_U8_S16_S16(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    Iterator input1(in1, window.broadcast_if_dimension_le_one(in1->info()->tensor_shape()));
    Iterator input2(in2, window.broadcast_if_dimension_le_one(in2->info()->tensor_shape()));
    Iterator output(out, window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        const uint8x16_t bv_0 = vld1q_u8(input1.ptr());
        int16x8_t        a1_0 = vld1q_s16(reinterpret_cast<const int16_t *>(input2.ptr()));
        int16x8_t        a2_0 = vld1q_s16(reinterpret_cast<const int16_t *>(input2.ptr()) + 8);

        a1_0 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(bv_0))), a1_0);
        a2_0 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(bv_0))), a2_0);

        vst1q_s16(reinterpret_cast<int16_t *>(output.ptr()), a1_0);
        vst1q_s16(reinterpret_cast<int16_t *>(output.ptr()) + 8, a2_0);
    },
    input1, input2, output);
}

void sub_saturate_U8_S16_S16(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    Iterator input1(in1, window.broadcast_if_dimension_le_one(in1->info()->tensor_shape()));
    Iterator input2(in2, window.broadcast_if_dimension_le_one(in2->info()->tensor_shape()));
    Iterator output(out, window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        const uint8x16_t bv_0 = vld1q_u8(input1.ptr());
        int16x8_t        a1_0 = vld1q_s16(reinterpret_cast<const int16_t *>(input2.ptr()));
        int16x8_t        a2_0 = vld1q_s16(reinterpret_cast<const int16_t *>(input2.ptr()) + 8);

        a1_0 = vqsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(bv_0))), a1_0);
        a2_0 = vqsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(bv_0))), a2_0);

        vst1q_s16(reinterpret_cast<int16_t *>(output.ptr()), a1_0);
        vst1q_s16(reinterpret_cast<int16_t *>(output.ptr()) + 8, a2_0);
    },
    input1, input2, output);
}

void sub_wrap_U8_U8_S16(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    Iterator input1(in1, window.broadcast_if_dimension_le_one(in1->info()->tensor_shape()));
    Iterator input2(in2, window.broadcast_if_dimension_le_one(in2->info()->tensor_shape()));
    Iterator output(out, window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        const uint8x16_t av_0 = vld1q_u8(input1.ptr());
        const uint8x16_t bv_0 = vld1q_u8(input2.ptr());

        const int16x8_t a1_0 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(av_0))),
                                         vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(bv_0))));
        const int16x8_t a2_0 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(av_0))),
                                         vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(bv_0))));

        vst1q_s16(reinterpret_cast<int16_t *>(output.ptr()), a1_0);
        vst1q_s16(reinterpret_cast<int16_t *>(output.ptr()) + 8, a2_0);
    },
    input1, input2, output);
}

void sub_saturate_U8_U8_S16(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    Iterator input1(in1, window.broadcast_if_dimension_le_one(in1->info()->tensor_shape()));
    Iterator input2(in2, window.broadcast_if_dimension_le_one(in2->info()->tensor_shape()));
    Iterator output(out, window);

    execute_window_loop(window, [&](const Coordinates &)
    {
        const uint8x16_t av_0 = vld1q_u8(input1.ptr());
        const uint8x16_t bv_0 = vld1q_u8(input2.ptr());

        const int16x8_t a1_0 = vqsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(av_0))),
                                          vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(bv_0))));
        const int16x8_t a2_0 = vqsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(av_0))),
                                          vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(bv_0))));

        vst1q_s16(reinterpret_cast<int16_t *>(output.ptr()), a1_0);
        vst1q_s16(reinterpret_cast<int16_t *>(output.ptr()) + 8, a2_0);
    },
    input1, input2, output);
}

inline Status validate_arguments(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output, ConvertPolicy policy)
{
    ARM_COMPUTE_UNUSED(policy);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(&input1);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input1, 1, DataType::U8, DataType::QASYMM8, DataType::S16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input2, 1, DataType::U8, DataType::QASYMM8, DataType::S16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&output, 1, DataType::U8, DataType::QASYMM8, DataType::S16, DataType::F16, DataType::F32);

    const TensorShape out_shape = TensorShape::broadcast_shape(input1.tensor_shape(), input2.tensor_shape());
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(
        !(input1.data_type() == DataType::U8 && input2.data_type() == DataType::U8)
        && !(input1.data_type() == DataType::QASYMM8 && input2.data_type() == DataType::QASYMM8)
        && !(input1.data_type() == DataType::U8 && input2.data_type() == DataType::U8)
        && !(input1.data_type() == DataType::U8 && input2.data_type() == DataType::S16)
        && !(input1.data_type() == DataType::S16 && input2.data_type() == DataType::U8)
        && !(input1.data_type() == DataType::S16 && input2.data_type() == DataType::S16)
        && !(input1.data_type() == DataType::F32 && input2.data_type() == DataType::F32)
        && !(input1.data_type() == DataType::F16 && input2.data_type() == DataType::F16),
        "You called subtract with the wrong image formats");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(
        input1.data_type() == DataType::QASYMM8 && input2.data_type() == DataType::QASYMM8 && policy == ConvertPolicy::WRAP,
        "Convert policy cannot be WRAP if datatype is QASYMM8");

    // Validate in case of configured output
    if(output.total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(
            !(input1.data_type() == DataType::U8 && input2.data_type() == DataType::U8 && output.data_type() == DataType::U8)
            && !(input1.data_type() == DataType::QASYMM8 && input2.data_type() == DataType::QASYMM8 && output.data_type() == DataType::QASYMM8)
            && !(input1.data_type() == DataType::U8 && input2.data_type() == DataType::U8 && output.data_type() == DataType::S16)
            && !(input1.data_type() == DataType::U8 && input2.data_type() == DataType::S16 && output.data_type() == DataType::S16)
            && !(input1.data_type() == DataType::S16 && input2.data_type() == DataType::U8 && output.data_type() == DataType::S16)
            && !(input1.data_type() == DataType::S16 && input2.data_type() == DataType::S16 && output.data_type() == DataType::S16)
            && !(input1.data_type() == DataType::F32 && input2.data_type() == DataType::F32 && output.data_type() == DataType::F32)
            && !(input1.data_type() == DataType::F16 && input2.data_type() == DataType::F16 && output.data_type() == DataType::F16),
            "You called subtract with the wrong image formats");

        ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(out_shape, output.tensor_shape(), 0),
                                        "Wrong shape for output");
    }
    return Status{};
}

inline std::pair<Status, Window> validate_and_configure_window(ITensorInfo &input1, ITensorInfo &input2, ITensorInfo &output)
{
    const std::pair<TensorShape, ValidRegion> broadcast_pair = ITensorInfo::broadcast_shape_and_valid_region(input1, input2);
    const TensorShape &out_shape    = broadcast_pair.first;
    const ValidRegion &valid_region = broadcast_pair.second;

    // Auto initialize output if not initialized
    {
        set_shape_if_empty(output, out_shape);

        if(input1.data_type() == DataType::S16 || input2.data_type() == DataType::S16)
        {
            set_format_if_unknown(output, Format::S16);
        }
        else if(input1.data_type() == DataType::F16 && input2.data_type() == DataType::F16)
        {
            set_format_if_unknown(output, Format::F16);
        }
        else if(input1.data_type() == DataType::F32 || input2.data_type() == DataType::F32)
        {
            set_format_if_unknown(output, Format::F32);
        }
    }

    Window win        = calculate_max_window(valid_region, Steps(num_elems_processed_per_iteration));
    Window win_input1 = win.broadcast_if_dimension_le_one(input1);
    Window win_input2 = win.broadcast_if_dimension_le_one(input2);

    AccessWindowHorizontal input1_access(&input1, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal input2_access(&input2, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(&output, 0, num_elems_processed_per_iteration);

    bool window_changed = update_window_and_padding(win_input1, input1_access)
                          || update_window_and_padding(win_input2, input2_access)
                          || update_window_and_padding(win, output_access);

    output_access.set_valid_region(win, valid_region);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

NEArithmeticSubtractionKernel::NEArithmeticSubtractionKernel()
    : _func(nullptr), _input1(nullptr), _input2(nullptr), _output(nullptr)
{
}

void NEArithmeticSubtractionKernel::configure(const ITensor *input1, const ITensor *input2, ITensor *output, ConvertPolicy policy)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*input1->info(), *input2->info(), *output->info(), policy));

    // Configure kernel window
    auto win_config = validate_and_configure_window(*input1->info(), *input2->info(), *output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);

    static std::map<std::string, NEArithmeticSubtractionKernel::SubFunction *> map_function =
    {
        { "sub_wrap_U8_U8_U8", &sub_wrap_U8_U8_U8 },
        { "sub_wrap_U8_U8_S16", &sub_wrap_U8_U8_S16 },
        { "sub_saturate_U8_U8_U8", &sub_saturate_U8_U8_U8 },
        { "sub_saturate_U8_U8_S16", &sub_saturate_U8_U8_S16 },
        { "sub_saturate_QASYMM8_QASYMM8_QASYMM8", &sub_saturate_QAYSMM8_QAYSMM8_QAYSMM8 },
        { "sub_wrap_U8_S16_S16", &sub_wrap_U8_S16_S16 },
        { "sub_wrap_S16_U8_S16", &sub_wrap_S16_U8_S16 },
        { "sub_saturate_U8_S16_S16", &sub_saturate_U8_S16_S16 },
        { "sub_saturate_S16_U8_S16", &sub_saturate_S16_U8_S16 },
        { "sub_wrap_S16_S16_S16", &sub_wrap_S16_S16_S16 },
        { "sub_saturate_S16_S16_S16", &sub_saturate_S16_S16_S16 },
        { "sub_wrap_F32_F32_F32", &sub_F32_F32_F32 },
        { "sub_saturate_F32_F32_F32", &sub_F32_F32_F32 },
        { "sub_wrap_F16_F16_F16", &sub_F16_F16_F16 },
        { "sub_saturate_F16_F16_F16", &sub_F16_F16_F16 },
    };

    _input1 = input1;
    _input2 = input2;
    _output = output;

    std::string function_to_call("sub_");
    function_to_call += policy == ConvertPolicy::WRAP ? "wrap_" : "saturate_";
    function_to_call += string_from_data_type(input1->info()->data_type()) + "_";
    function_to_call += string_from_data_type(input2->info()->data_type()) + "_";
    function_to_call += string_from_data_type(output->info()->data_type());

    auto it = map_function.find(function_to_call);

    if(it != map_function.end())
    {
        _func = it->second;
    }

    INEKernel::configure(win_config.second);
}

Status NEArithmeticSubtractionKernel::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input1, input2, output);

    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*input1, *input2, *output, policy));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(*input1->clone(), *input2->clone(), *output->clone()).first);

    return Status{};
}

void NEArithmeticSubtractionKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (*_func)(_input1, _input2, _output, window);
}

BorderSize NEArithmeticSubtractionKernel::border_size() const
{
    const unsigned int replicateSize = _output->info()->dimension(0) - std::min(_input1->info()->dimension(0), _input2->info()->dimension(0));
    const unsigned int border        = std::min<unsigned int>(num_elems_processed_per_iteration - 1U, replicateSize);
    return BorderSize{ 0, border, 0, 0 };
}