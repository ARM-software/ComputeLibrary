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
#include "arm_compute/core/NEON/kernels/NEArithmeticAdditionKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
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
void add_wrap_QS8_QS8_QS8(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    Iterator input1(in1, window);
    Iterator input2(in2, window);
    Iterator output(out, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const qint8x16_t a = vld1q_qs8(reinterpret_cast<const qint8_t *>(input1.ptr()));
        const qint8x16_t b = vld1q_qs8(reinterpret_cast<const qint8_t *>(input2.ptr()));

        vst1q_qs8(reinterpret_cast<qint8_t *>(output.ptr()), vaddq_qs8(a, b));
    },
    input1, input2, output);
}

void add_saturate_QS8_QS8_QS8(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    Iterator input1(in1, window);
    Iterator input2(in2, window);
    Iterator output(out, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const qint8x16_t a = vld1q_qs8(reinterpret_cast<const qint8_t *>(input1.ptr()));
        const qint8x16_t b = vld1q_qs8(reinterpret_cast<const qint8_t *>(input2.ptr()));

        vst1q_qs8(reinterpret_cast<qint8_t *>(output.ptr()), vqaddq_qs8(a, b));
    },
    input1, input2, output);
}

void add_wrap_U8_U8_U8(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    Iterator input1(in1, window);
    Iterator input2(in2, window);
    Iterator output(out, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        vst1q_u8(output.ptr(), vaddq_u8(vld1q_u8(input1.ptr()), vld1q_u8(input2.ptr())));
    },
    input1, input2, output);
}

void add_saturate_U8_U8_U8(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    Iterator input1(in1, window);
    Iterator input2(in2, window);
    Iterator output(out, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        vst1q_u8(output.ptr(), vqaddq_u8(vld1q_u8(input1.ptr()), vld1q_u8(input2.ptr())));
    },
    input1, input2, output);
}

inline int16x8x2_t vadd2q_s16(const int16x8x2_t &a, const int16x8x2_t &b)
{
    const int16x8x2_t res =
    {
        {
            vaddq_s16(a.val[0], b.val[0]),
            vaddq_s16(a.val[1], b.val[1])
        }
    };

    return res;
}

inline float32x4x4_t vadd4q_f32(const float32x4x4_t &a, const float32x4x4_t &b)
{
    const float32x4x4_t res =
    {
        {
            vaddq_f32(a.val[0], b.val[0]),
            vaddq_f32(a.val[1], b.val[1]),
            vaddq_f32(a.val[2], b.val[2]),
            vaddq_f32(a.val[3], b.val[3])
        }
    };

    return res;
}

inline int16x8x2_t vqadd2q_s16(const int16x8x2_t &a, const int16x8x2_t &b)
{
    const int16x8x2_t res =
    {
        {
            vqaddq_s16(a.val[0], b.val[0]),
            vqaddq_s16(a.val[1], b.val[1])
        }
    };

    return res;
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
inline float16x8x2_t vadd2q_f16(const float16x8x2_t &a, const float16x8x2_t &b)
{
    const float16x8x2_t res =
    {
        {
            vaddq_f16(a.val[0], b.val[0]),
            vaddq_f16(a.val[1], b.val[1])
        }
    };

    return res;
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

void add_F16_F16_F16(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    Iterator input1(in1, window);
    Iterator input2(in2, window);
    Iterator output(out, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const float16x8x2_t a = vld2q_f16(reinterpret_cast<const float16_t *>(input1.ptr()));
        const float16x8x2_t b = vld2q_f16(reinterpret_cast<const float16_t *>(input2.ptr()));

        vst2q_f16(reinterpret_cast<float16_t *>(output.ptr()), vadd2q_f16(a, b));
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

void add_F32_F32_F32(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    Iterator input1(in1, window);
    Iterator input2(in2, window);
    Iterator output(out, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const float32x4x4_t a = vld4q_f32(reinterpret_cast<const float *>(input1.ptr()));
        const float32x4x4_t b = vld4q_f32(reinterpret_cast<const float *>(input2.ptr()));

        vst4q_f32(reinterpret_cast<float *>(output.ptr()), vadd4q_f32(a, b));
    },
    input1, input2, output);
}

void add_wrap_S16_S16_S16(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    Iterator input1(in1, window);
    Iterator input2(in2, window);
    Iterator output(out, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const int16x8x2_t a = vld2q_s16(reinterpret_cast<const int16_t *>(input1.ptr()));
        const int16x8x2_t b = vld2q_s16(reinterpret_cast<const int16_t *>(input2.ptr()));

        vst2q_s16(reinterpret_cast<int16_t *>(output.ptr()), vadd2q_s16(a, b));
    },
    input1, input2, output);
}

void add_saturate_S16_S16_S16(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    Iterator input1(in1, window);
    Iterator input2(in2, window);
    Iterator output(out, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const int16x8x2_t a = vld2q_s16(reinterpret_cast<const int16_t *>(input1.ptr()));
        const int16x8x2_t b = vld2q_s16(reinterpret_cast<const int16_t *>(input2.ptr()));

        vst2q_s16(reinterpret_cast<int16_t *>(output.ptr()), vqadd2q_s16(a, b));
    },
    input1, input2, output);
}

void add_wrap_S16_U8_S16(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    Iterator input1(in1, window);
    Iterator input2(in2, window);
    Iterator output(out, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const int16x8x2_t a =
        {
            {
                vld1q_s16(reinterpret_cast<const int16_t *>(input1.ptr())),
                vld1q_s16(reinterpret_cast<const int16_t *>(input1.ptr()) + 8)
            }
        };
        const uint8x16_t b = vld1q_u8(input2.ptr());

        vst1q_s16(reinterpret_cast<int16_t *>(output.ptr()), vaddq_s16(a.val[0], vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b)))));
        vst1q_s16(reinterpret_cast<int16_t *>(output.ptr()) + 8, vaddq_s16(a.val[1], vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(b)))));
    },
    input1, input2, output);
}

void add_saturate_S16_U8_S16(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    Iterator input1(in1, window);
    Iterator input2(in2, window);
    Iterator output(out, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const int16x8x2_t a =
        {
            {
                vld1q_s16(reinterpret_cast<const int16_t *>(input1.ptr())),
                vld1q_s16(reinterpret_cast<const int16_t *>(input1.ptr()) + 8)
            }
        };
        const uint8x16_t b = vld1q_u8(input2.ptr());

        vst1q_s16(reinterpret_cast<int16_t *>(output.ptr()), vqaddq_s16(a.val[0], vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b)))));
        vst1q_s16(reinterpret_cast<int16_t *>(output.ptr()) + 8, vqaddq_s16(a.val[1], vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(b)))));
    },
    input1, input2, output);
}

inline void add_wrap_U8_S16_S16(const ITensor *input1, const ITensor *input2, ITensor *output, const Window &window)
{
    //Simply swap the two input buffers:
    add_wrap_S16_U8_S16(input2, input1, output, window);
}

inline void add_saturate_U8_S16_S16(const ITensor *input1, const ITensor *input2, ITensor *output, const Window &window)
{
    //Simply swap the two input buffers:
    add_saturate_S16_U8_S16(input2, input1, output, window);
}

void add_wrap_U8_U8_S16(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    Iterator input1(in1, window);
    Iterator input2(in2, window);
    Iterator output(out, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const uint8x16_t a = vld1q_u8(input1.ptr());
        const uint8x16_t b = vld1q_u8(input2.ptr());

        const int16x8x2_t a_s16 =
        {
            {
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a))),
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(a)))
            }
        };

        const int16x8x2_t b_s16 =
        {
            {
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b))),
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(b)))
            }
        };

        vst1q_s16(reinterpret_cast<int16_t *>(output.ptr()), vaddq_s16(a_s16.val[0], b_s16.val[0]));
        vst1q_s16(reinterpret_cast<int16_t *>(output.ptr()) + 8, vaddq_s16(a_s16.val[1], b_s16.val[1]));
    },
    input1, input2, output);
}

void add_saturate_U8_U8_S16(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window)
{
    Iterator input1(in1, window);
    Iterator input2(in2, window);
    Iterator output(out, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const uint8x16_t a = vld1q_u8(input1.ptr());
        const uint8x16_t b = vld1q_u8(input2.ptr());

        const int16x8x2_t a_s16 =
        {
            {
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a))),
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(a)))
            }
        };

        const int16x8x2_t b_s16 =
        {
            {
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b))),
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(b)))
            }
        };

        vst1q_s16(reinterpret_cast<int16_t *>(output.ptr()), vqaddq_s16(a_s16.val[0], b_s16.val[0]));
        vst1q_s16(reinterpret_cast<int16_t *>(output.ptr()) + 8, vqaddq_s16(a_s16.val[1], b_s16.val[1]));
    },
    input1, input2, output);
}

inline Status validate_arguments(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy)
{
    ARM_COMPUTE_UNUSED(policy);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input1, input2, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input1, 1, DataType::QS8, DataType::U8, DataType::QS16, DataType::S16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input2, 1, DataType::QS8, DataType::U8, DataType::QS16, DataType::S16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::QS8, DataType::U8, DataType::QS16, DataType::S16, DataType::F16, DataType::F32);

    if(is_data_type_fixed_point(input1->data_type()) || is_data_type_fixed_point(input2->data_type()) || is_data_type_fixed_point(output->data_type()))
    {
        // Check that all data types are the same and all fixed-point positions are the same
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT(input1, input2, output);
    }

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(
        !(input1->data_type() == DataType::QS8 && input2->data_type() == DataType::QS8 && output->data_type() == DataType::QS8)
        && !(input1->data_type() == DataType::U8 && input2->data_type() == DataType::U8 && output->data_type() == DataType::U8)
        && !(input1->data_type() == DataType::U8 && input2->data_type() == DataType::U8 && output->data_type() == DataType::S16)
        && !(input1->data_type() == DataType::U8 && input2->data_type() == DataType::S16 && output->data_type() == DataType::S16)
        && !(input1->data_type() == DataType::S16 && input2->data_type() == DataType::U8 && output->data_type() == DataType::S16)
        && !(input1->data_type() == DataType::QS16 && input2->data_type() == DataType::QS16 && output->data_type() == DataType::QS16)
        && !(input1->data_type() == DataType::S16 && input2->data_type() == DataType::S16 && output->data_type() == DataType::S16)
        && !(input1->data_type() == DataType::F32 && input2->data_type() == DataType::F32 && output->data_type() == DataType::F32)
        && !(input1->data_type() == DataType::F16 && input2->data_type() == DataType::F16 && output->data_type() == DataType::F16),
        "You called addition with the wrong image formats");

    return Status{};
}

inline std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output)
{
    constexpr unsigned int num_elems_processed_per_iteration = 16;

    // Configure kernel window
    Window                 win = calculate_max_window(*input1, Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);

    bool window_changed = update_window_and_padding(win,
                                                    AccessWindowHorizontal(input1, 0, num_elems_processed_per_iteration),
                                                    AccessWindowHorizontal(input2, 0, num_elems_processed_per_iteration),
                                                    output_access);

    ValidRegion valid_region = intersect_valid_regions(input1->valid_region(),
                                                       input2->valid_region());

    output_access.set_valid_region(win, valid_region);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

NEArithmeticAdditionKernel::NEArithmeticAdditionKernel()
    : _func(nullptr), _input1(nullptr), _input2(nullptr), _output(nullptr)
{
}

void NEArithmeticAdditionKernel::configure(const ITensor *input1, const ITensor *input2, ITensor *output, ConvertPolicy policy)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, output);

    // Auto initialize output if not initialized
    {
        set_shape_if_empty(*output->info(), input1->info()->tensor_shape());

        if(input1->info()->data_type() == DataType::S16 || input2->info()->data_type() == DataType::S16)
        {
            set_format_if_unknown(*output->info(), Format::S16);
        }
        else if(input1->info()->data_type() == DataType::F16 || input2->info()->data_type() == DataType::F16)
        {
            set_format_if_unknown(*output->info(), Format::F16);
        }
        else if(input1->info()->data_type() == DataType::F32 || input2->info()->data_type() == DataType::F32)
        {
            set_format_if_unknown(*output->info(), Format::F32);
        }
    }

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input1->info(), input2->info(), output->info(), policy));

    static std::map<std::string, AddFunction *> map_function =
    {
        { "add_wrap_QS8_QS8_QS8", &add_wrap_QS8_QS8_QS8 },
        { "add_saturate_QS8_QS8_QS8", &add_saturate_QS8_QS8_QS8 },
        { "add_wrap_U8_U8_U8", &add_wrap_U8_U8_U8 },
        { "add_saturate_U8_U8_U8", &add_saturate_U8_U8_U8 },
        { "add_wrap_S16_U8_S16", &add_wrap_S16_U8_S16 },
        { "add_saturate_S16_U8_S16", &add_saturate_S16_U8_S16 },
        { "add_wrap_U8_S16_S16", &add_wrap_U8_S16_S16 },
        { "add_saturate_U8_S16_S16", &add_saturate_U8_S16_S16 },
        { "add_wrap_U8_U8_S16", &add_wrap_U8_U8_S16 },
        { "add_saturate_U8_U8_S16", &add_saturate_U8_U8_S16 },
        { "add_wrap_QS16_QS16_QS16", &add_wrap_S16_S16_S16 },
        { "add_saturate_QS16_QS16_QS16", &add_saturate_S16_S16_S16 },
        { "add_wrap_S16_S16_S16", &add_wrap_S16_S16_S16 },
        { "add_saturate_S16_S16_S16", &add_saturate_S16_S16_S16 },
        { "add_wrap_F32_F32_F32", &add_F32_F32_F32 },
        { "add_saturate_F32_F32_F32", &add_F32_F32_F32 },
        { "add_wrap_F16_F16_F16", &add_F16_F16_F16 },
        { "add_saturate_F16_F16_F16", &add_F16_F16_F16 },
    };

    _input1 = input1;
    _input2 = input2;
    _output = output;

    std::string function_to_call("add_");
    function_to_call += policy == ConvertPolicy::WRAP ? "wrap_" : "saturate_";
    function_to_call += string_from_data_type(input1->info()->data_type()) + "_";
    function_to_call += string_from_data_type(input2->info()->data_type()) + "_";
    function_to_call += string_from_data_type(output->info()->data_type());

    auto it = map_function.find(function_to_call);

    if(it != map_function.end())
    {
        _func = it->second;
    }

    // Configure kernel window
    auto win_config = validate_and_configure_window(input1->info(), input2->info(), output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NEArithmeticAdditionKernel::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input1, input2, output, policy));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input1->clone().get(), input2->clone().get(), output->clone().get()).first);

    return Status{};
}

void NEArithmeticAdditionKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (*_func)(_input1, _input2, _output, window);
}
