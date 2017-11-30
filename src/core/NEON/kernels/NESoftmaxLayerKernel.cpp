/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NESoftmaxLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/NEON/NEMath.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <algorithm>
#include <arm_neon.h>
#include <cfloat>

using namespace arm_compute;

namespace
{
Status validate_arguments_logits_1d_max(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        // Softmax across the x dimension
        TensorShape output_shape{ input->tensor_shape() };
        output_shape.set(0, 1);

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT_POSITION(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), output_shape);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window_logits_1d_max(ITensorInfo *input, ITensorInfo *output)
{
    // Configure kernel window
    constexpr unsigned int num_elems_written_per_row = 1;
    const int              input_width               = input->valid_region().shape.x();

    unsigned int           num_elems_processed_per_iteration = 16 / data_size_from_type(input->data_type());
    Window                 win                               = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
    bool                   window_changed = false;

    if(output->total_size() != 0)
    {
        AccessWindowHorizontal output_access(output, 0, num_elems_written_per_row, 1.f / input_width);
        window_changed = update_window_and_padding(win, input_access, output_access);
        output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));
    }
    else
    {
        window_changed = update_window_and_padding(win, input_access);
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

Status validate_arguments_logits_1d_shift_exp_sum(const ITensorInfo *input, const ITensorInfo *max, const ITensorInfo *output, const ITensorInfo *sum, float beta)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, max, sum, output);
    ARM_COMPUTE_RETURN_ERROR_ON((beta != 1.0f) && is_data_type_fixed_point(input->data_type()));
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT_POSITION(input, output);
    }

    // Checks performed when sum is configured
    if(sum->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, max, sum);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(max, sum);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT_POSITION(input, max, sum);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window_logits_1d_shift_exp_sum(ITensorInfo *input, ITensorInfo *max, ITensorInfo *output, ITensorInfo *sum)
{
    unsigned int num_elems_processed_per_iteration = input->valid_region().shape.x();

    // Configure kernel window
    Window                 win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal max_access(max, 0, 1);
    AccessWindowHorizontal sum_access(sum, 0, 1);
    bool                   window_changed = false;

    if(output->total_size() != 0)
    {
        AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
        window_changed = update_window_and_padding(win, input_access, max_access, output_access, sum_access);
        output_access.set_valid_region(win, input->valid_region());
    }
    else
    {
        window_changed = update_window_and_padding(win, input_access, max_access, sum_access);
    }

    sum_access.set_valid_region(win, ValidRegion(Coordinates(), sum->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

Status validate_arguments_logits_1d_norm(const ITensorInfo *input, const ITensorInfo *sum, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, sum, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QS16, DataType::S32, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, sum);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT_POSITION(input, sum);

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT_POSITION(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window_logits_1d_norm(ITensorInfo *input, ITensorInfo *sum, ITensorInfo *output)
{
    // Configure kernel window
    unsigned int num_elems_processed_per_iteration = 16 / data_size_from_type(input->data_type());
    Window       win                               = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
    AccessWindowStatic     sum_access(sum, 0, 0, 1, sum->dimension(1));
    bool                   window_changed = false;

    if(output->total_size() != 0)
    {
        AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);

        window_changed = update_window_and_padding(win, input_access, sum_access, output_access);

        output_access.set_valid_region(win, input->valid_region());
    }
    else
    {
        window_changed = update_window_and_padding(win, input_access, sum_access);
    }
    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

void logits_1d_max_qs8(const ITensor *in, ITensor *out, const Window &window)
{
    Window in_slice = window.first_slice_window_1D();

    Window window_max(window);
    window_max.set(Window::DimX, Window::Dimension(0, 0, 0));
    Window max_slice = window_max.first_slice_window_1D();

    do
    {
        Iterator input(in, in_slice);
        Iterator output(out, max_slice);

        qint8x16_t vec_max = vdupq_n_s8(std::numeric_limits<qint8_t>::lowest());

        execute_window_loop(in_slice, [&](const Coordinates & id)
        {
            const auto       in_ptr        = reinterpret_cast<const qint8_t *>(input.ptr());
            const qint8x16_t current_value = vld1q_qs8(in_ptr);
            vec_max                        = vmaxq_qs8(vec_max, current_value);
        },
        input);

        qint8x8_t carry_max = vpmax_qs8(vget_high_s8(vec_max), vget_low_s8(vec_max));
        carry_max           = vpmax_qs8(carry_max, carry_max);
        carry_max           = vpmax_qs8(carry_max, carry_max);
        carry_max           = vpmax_qs8(carry_max, carry_max);

        *(reinterpret_cast<qint8_t *>(output.ptr())) = vget_lane_s8(carry_max, 0);
    }
    while(window.slide_window_slice_1D(in_slice) && window.slide_window_slice_1D(max_slice));
}
void logits_1d_max_qs16(const ITensor *in, ITensor *out, const Window &window)
{
    Window in_slice = window.first_slice_window_1D();

    Window window_max(window);
    window_max.set(Window::DimX, Window::Dimension(0, 0, 0));
    Window max_slice = window_max.first_slice_window_1D();

    do
    {
        Iterator input(in, in_slice);
        Iterator output(out, max_slice);

        qint16x8_t vec_max = vdupq_n_qs16(std::numeric_limits<qint16_t>::lowest());

        execute_window_loop(in_slice, [&](const Coordinates & id)
        {
            const auto       in_ptr        = reinterpret_cast<const qint16_t *>(input.ptr());
            const qint16x8_t current_value = vld1q_qs16(in_ptr);
            vec_max                        = vmaxq_qs16(vec_max, current_value);
        },
        input);

        qint16x4_t carry_max = vpmax_qs16(vget_high_qs16(vec_max), vget_low_qs16(vec_max));
        carry_max            = vpmax_qs16(carry_max, carry_max);
        carry_max            = vpmax_qs16(carry_max, carry_max);

        *(reinterpret_cast<qint16_t *>(output.ptr())) = vget_lane_s16(carry_max, 0);
    }
    while(window.slide_window_slice_1D(in_slice) && window.slide_window_slice_1D(max_slice));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
void logits_1d_max_f16(const ITensor *in, ITensor *out, const Window &window)
{
    Window in_slice = window.first_slice_window_1D();

    Window window_max(window);
    window_max.set(Window::DimX, Window::Dimension(0, 0, 0));
    Window max_slice = window_max.first_slice_window_1D();

    do
    {
        Iterator input(in, in_slice);
        Iterator output(out, max_slice);

        float16x8_t vec_max = vdupq_n_f16(std::numeric_limits<float16_t>::lowest());

        execute_window_loop(in_slice, [&](const Coordinates & id)
        {
            const auto        in_ptr        = reinterpret_cast<const float16_t *>(input.ptr());
            const float16x8_t current_value = vld1q_f16(in_ptr);
            vec_max                         = vmaxq_f16(vec_max, current_value);
        },
        input);

        float16x4_t carry_max = vpmax_f16(vget_high_f16(vec_max), vget_low_f16(vec_max));
        carry_max             = vpmax_f16(carry_max, carry_max);
        carry_max             = vpmax_f16(carry_max, carry_max);

        *(reinterpret_cast<float16_t *>(output.ptr())) = vget_lane_f16(carry_max, 0);
    }
    while(window.slide_window_slice_1D(in_slice) && window.slide_window_slice_1D(max_slice));
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

void logits_1d_max_f32(const ITensor *in, ITensor *out, const Window &window)
{
    Window in_slice = window.first_slice_window_1D();

    Window window_max(window);
    window_max.set(Window::DimX, Window::Dimension(0, 0, 0));
    Window max_slice = window_max.first_slice_window_1D();

    do
    {
        Iterator input(in, in_slice);
        Iterator output(out, max_slice);

        float32x4_t vec_max = vdupq_n_f32(-FLT_MAX);

        execute_window_loop(in_slice, [&](const Coordinates & id)
        {
            const auto        in_ptr        = reinterpret_cast<const float *>(input.ptr());
            const float32x4_t current_value = vld1q_f32(in_ptr);
            vec_max                         = vmaxq_f32(vec_max, current_value);
        },
        input);

        float32x2_t carry_max = vpmax_f32(vget_high_f32(vec_max), vget_low_f32(vec_max));
        carry_max             = vpmax_f32(carry_max, carry_max);

        *(reinterpret_cast<float *>(output.ptr())) = vget_lane_f32(carry_max, 0);
    }
    while(window.slide_window_slice_1D(in_slice) && window.slide_window_slice_1D(max_slice));
}
} // namespace

NELogits1DMaxKernel::NELogits1DMaxKernel()
    : _func(nullptr), _border_size()
{
}

BorderSize NELogits1DMaxKernel::border_size() const
{
    return _border_size;
}

void NELogits1DMaxKernel::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Softmax across the x dimension
    TensorShape output_shape{ input->info()->tensor_shape() };
    output_shape.set(0, 1);

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type(), input->info()->fixed_point_position());

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_logits_1d_max(input->info(), output->info()));

    const int    input_width                       = input->info()->valid_region().shape.x();
    unsigned int num_elems_processed_per_iteration = 16 / data_size_from_type(input->info()->data_type());

    switch(input->info()->data_type())
    {
        case DataType::QS8:
            _func = &logits_1d_max_qs8;
            break;
        case DataType::QS16:
            _func = &logits_1d_max_qs16;
            break;
        case DataType::F32:
            _func = &logits_1d_max_f32;
            break;
        case DataType::F16:
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            _func = &logits_1d_max_f16;
            break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        default:
            ARM_COMPUTE_ERROR("Unsupported data type.");
    }

    _input       = input;
    _output      = output;
    _border_size = BorderSize(0, num_elems_processed_per_iteration - (input_width % num_elems_processed_per_iteration), 0, 0);

    // Configure kernel window
    auto win_config = validate_and_configure_window_logits_1d_max(input->info(), output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NELogits1DMaxKernel::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_logits_1d_max(input, output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_logits_1d_max(input->clone().get(), output->clone().get()).first);

    return Status{};
}

void NELogits1DMaxKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (*_func)(_input, _output, window);
}

namespace
{
void logits_1d_shift_exp_sum_qs8(const ITensor *in, const ITensor *max, ITensor *out, ITensor *sum, const Window &window, float beta)
{
    ARM_COMPUTE_UNUSED(beta);

    Window window_max(window);
    window_max.set(Window::DimX, Window::Dimension(0, 0, 0));

    Window max_slice = window_max.first_slice_window_1D();
    Window in_slice  = window.first_slice_window_1D();

    constexpr int step                 = 8;
    const int     long_steps           = in->info()->valid_region().shape.x() / step;
    const int     small_steps          = in->info()->valid_region().shape.x() % step;
    const int     fixed_point_position = in->info()->fixed_point_position();

    do
    {
        Iterator input(in, in_slice);
        Iterator exp(out, in_slice);
        Iterator _max(max, max_slice);
        Iterator _sum(sum, max_slice);

        // Get pointers
        auto in_ptr  = reinterpret_cast<const qint8_t *>(input.ptr());
        auto exp_ptr = reinterpret_cast<qint8_t *>(exp.ptr());

        // Init sum to zero
        qint16x8_t vec_sum_value = vdupq_n_qs16(0);

        // Get max value
        const auto      max_ptr = reinterpret_cast<const qint8_t *>(_max.ptr());
        const qint8x8_t vec_max = vdup_n_qs8(*max_ptr);

        // Run neon loop
        for(int i = 0; i < long_steps; ++i)
        {
            qint8x8_t vec_elements = vld1_qs8(in_ptr);
            vec_elements           = vqsub_qs8(vec_elements, vec_max);
            vec_elements           = vqexp_qs8(vec_elements, fixed_point_position);

            vst1_qs8(exp_ptr, vec_elements);
            vec_sum_value = vqaddq_qs16(vec_sum_value, vmovl_s8(vec_elements));

            in_ptr += step;
            exp_ptr += step;
        }
        // Reduce sum
        const qint16x4_t sum_red = vqadd_qs16(vget_low_s16(vec_sum_value), vget_high_s16(vec_sum_value));
        const qint16_t   sum0    = sqadd_qs16(vget_lane_s16(sum_red, 0), vget_lane_s16(sum_red, 1));
        const qint16_t   sum1    = sqadd_qs16(vget_lane_s16(sum_red, 2), vget_lane_s16(sum_red, 3));
        qint16_t         sum     = sqadd_qs16(sum0, sum1);

        // Run remaining elements
        for(int i = 0; i < small_steps; ++i)
        {
            qint8_t element = sqexp_qs8(sqsub_qs8(in_ptr[i], *max_ptr), fixed_point_position);
            exp_ptr[i]      = element;
            sum             = sqadd_qs16(sum, element);
        }

        *(reinterpret_cast<qint8_t *>(_sum.ptr())) = sqmovn_qs16(sum);
    }
    while(window.slide_window_slice_1D(in_slice) && window.slide_window_slice_1D(max_slice));
}
void logits_1d_shift_exp_sum_qs16(const ITensor *in, const ITensor *max, ITensor *out, ITensor *sum, const Window &window, float beta)
{
    ARM_COMPUTE_UNUSED(beta);

    Window window_max(window);
    window_max.set(Window::DimX, Window::Dimension(0, 0, 0));

    Window max_slice = window_max.first_slice_window_1D();
    Window in_slice  = window.first_slice_window_1D();

    constexpr int step                 = 4;
    const int     long_steps           = in->info()->valid_region().shape.x() / step;
    const int     small_steps          = in->info()->valid_region().shape.x() % step;
    const int     fixed_point_position = in->info()->fixed_point_position();

    do
    {
        Iterator input(in, in_slice);
        Iterator exp(out, in_slice);
        Iterator _max(max, max_slice);
        Iterator _sum(sum, max_slice);

        // Get pointers
        auto in_ptr  = reinterpret_cast<const qint16_t *>(input.ptr());
        auto exp_ptr = reinterpret_cast<qint16_t *>(exp.ptr());

        // Init sum to zero
        qint32x4_t vec_sum_value = vdupq_n_qs32(0);

        // Get max value
        const auto       max_ptr = reinterpret_cast<const qint16_t *>(_max.ptr());
        const qint16x4_t vec_max = vdup_n_qs16(*max_ptr);

        // Run neon loop
        for(int i = 0; i < long_steps; ++i)
        {
            qint16x4_t vec_elements = vld1_qs16(in_ptr);
            vec_elements            = vqsub_qs16(vec_elements, vec_max);
            vec_elements            = vqexp_qs16(vec_elements, fixed_point_position);

            vst1_qs16(exp_ptr, vec_elements);
            vec_sum_value = vqaddq_qs32(vec_sum_value, vmovl_s16(vec_elements));

            in_ptr += step;
            exp_ptr += step;
        }
        // Reduce sum
        qint32x2_t carry_addition = vqadd_qs32(vget_high_s32(vec_sum_value), vget_low_s32(vec_sum_value));
        qint32_t   sum            = vget_lane_s32(carry_addition, 0) + vget_lane_s32(carry_addition, 1);

        // Run remaining elements
        for(int i = 0; i < small_steps; ++i)
        {
            qint16_t element = sqexp_qs16(sqsub_qs16(in_ptr[i], *max_ptr), fixed_point_position);
            exp_ptr[i]       = element;
            sum              = sqadd_qs32(sum, element);
        }

        *(reinterpret_cast<qint16_t *>(_sum.ptr())) = sqmovn_qs32(sum);
    }
    while(window.slide_window_slice_1D(in_slice) && window.slide_window_slice_1D(max_slice));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
void logits_1d_shift_exp_sum_f16(const ITensor *in, const ITensor *max, ITensor *out, ITensor *sum, const Window &window, float beta)
{
    Window window_max(window);
    window_max.set(Window::DimX, Window::Dimension(0, 0, 0));

    Window max_slice = window_max.first_slice_window_1D();
    Window in_slice  = window.first_slice_window_1D();

    constexpr int step        = 8;
    const int     long_steps  = in->info()->valid_region().shape.x() / step;
    const int     small_steps = in->info()->valid_region().shape.x() % step;

    do
    {
        Iterator input(in, in_slice);
        Iterator exp(out, in_slice);
        Iterator _max(max, max_slice);
        Iterator _sum(sum, max_slice);

        // Get pointers
        auto in_ptr  = reinterpret_cast<const float16_t *>(input.ptr());
        auto exp_ptr = reinterpret_cast<float16_t *>(exp.ptr());

        // Init sum to zero
        float16x8_t vec_sum_value = vdupq_n_f16(0);

        // Get max value
        const auto        max_ptr = reinterpret_cast<const float16_t *>(_max.ptr());
        const float16x8_t vec_max = vdupq_n_f16(*max_ptr);

        // Run neon loop
        for(int i = 0; i < long_steps; ++i)
        {
            float16x8_t vec_elements = vld1q_f16(in_ptr);
            vec_elements             = vsubq_f16(vec_elements, vec_max);
            vec_elements             = vmulq_n_f16(vec_elements, beta);
            vec_elements             = vexpq_f16(vec_elements);

            vst1q_f16(exp_ptr, vec_elements);
            vec_sum_value = vaddq_f16(vec_sum_value, vec_elements);

            in_ptr += step;
            exp_ptr += step;
        }
        // Reduce sum
        const float16x4_t sum_red        = vadd_f16(vget_low_f16(vec_sum_value), vget_high_f16(vec_sum_value));
        const float16x4_t carry_addition = vpadd_f16(sum_red, sum_red);
        float16_t         sum            = vget_lane_f16(carry_addition, 0) + vget_lane_f16(carry_addition, 1);

        // Run remaining elements
        for(int i = 0; i < small_steps; ++i)
        {
            const float16_t element = std::exp(static_cast<float>(in_ptr[i] - *max_ptr) * beta);
            exp_ptr[i]              = element;
            sum += element;
        }
        *(reinterpret_cast<float16_t *>(_sum.ptr())) = sum;
    }
    while(window.slide_window_slice_1D(in_slice) && window.slide_window_slice_1D(max_slice));
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

void logits_1d_shift_exp_sum_f32(const ITensor *in, const ITensor *max, ITensor *out, ITensor *sum, const Window &window, float beta)
{
    Window window_max(window);
    window_max.set(Window::DimX, Window::Dimension(0, 0, 0));

    Window max_slice = window_max.first_slice_window_1D();
    Window in_slice  = window.first_slice_window_1D();

    constexpr int step        = 4;
    const int     long_steps  = in->info()->valid_region().shape.x() / step;
    const int     small_steps = in->info()->valid_region().shape.x() % step;

    do
    {
        Iterator input(in, in_slice);
        Iterator exp(out, in_slice);
        Iterator _max(max, max_slice);
        Iterator _sum(sum, max_slice);

        // Get pointers
        auto in_ptr  = reinterpret_cast<const float *>(input.ptr());
        auto exp_ptr = reinterpret_cast<float *>(exp.ptr());

        // Init sum to zero
        float32x4_t vec_sum_value = vdupq_n_f32(0.0f);

        // Get max value
        const auto        max_ptr = reinterpret_cast<const float *>(_max.ptr());
        const float32x4_t vec_max = vdupq_n_f32(*max_ptr);

        // Run neon loop
        for(int i = 0; i < long_steps; ++i)
        {
            float32x4_t vec_elements = vld1q_f32(in_ptr);
            vec_elements             = vsubq_f32(vec_elements, vec_max);
            vec_elements             = vmulq_n_f32(vec_elements, beta);
            vec_elements             = vexpq_f32(vec_elements);

            vst1q_f32(exp_ptr, vec_elements);
            vec_sum_value = vaddq_f32(vec_elements, vec_sum_value);

            in_ptr += step;
            exp_ptr += step;
        }

        // Reduce sum
        float32x2_t carry_addition = vpadd_f32(vget_high_f32(vec_sum_value), vget_low_f32(vec_sum_value));
        carry_addition             = vpadd_f32(carry_addition, carry_addition);
        float sum                  = vget_lane_f32(carry_addition, 0);

        // Run remaining elements
        for(int i = 0; i < small_steps; ++i)
        {
            float element = std::exp((in_ptr[i] - *max_ptr) * beta);
            exp_ptr[i]    = element;
            sum += element;
        }

        *(reinterpret_cast<float *>(_sum.ptr())) = sum;
    }
    while(window.slide_window_slice_1D(in_slice) && window.slide_window_slice_1D(max_slice));
}
} //namespace

NELogits1DShiftExpSumKernel::NELogits1DShiftExpSumKernel()
    : _func(nullptr), _input(nullptr), _max(nullptr), _output(nullptr), _sum(nullptr), _beta(1.0f)
{
}

void NELogits1DShiftExpSumKernel::configure(const ITensor *input, const ITensor *max, ITensor *output, ITensor *sum, float beta)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, max, sum, output);

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*sum->info(), max->info()->tensor_shape(), 1, input->info()->data_type(), input->info()->fixed_point_position());
    auto_init_if_empty(*output->info(), input->info()->tensor_shape(), 1, input->info()->data_type(), input->info()->fixed_point_position());

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_logits_1d_shift_exp_sum(input->info(), max->info(), output->info(), sum->info(), beta));

    switch(input->info()->data_type())
    {
        case DataType::QS8:
            _func = &logits_1d_shift_exp_sum_qs8;
            break;
        case DataType::QS16:
            _func = &logits_1d_shift_exp_sum_qs16;
            break;
        case DataType::F32:
            _func = &logits_1d_shift_exp_sum_f32;
            break;
        case DataType::F16:
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            _func = &logits_1d_shift_exp_sum_f16;
            break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        default:
            ARM_COMPUTE_ERROR("Unsupported data type.");
            break;
    }

    _input  = input;
    _max    = max;
    _output = output;
    _sum    = sum;
    _beta   = beta;

    // Configure kernel window
    auto win_config = validate_and_configure_window_logits_1d_shift_exp_sum(input->info(), max->info(), output->info(), sum->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NELogits1DShiftExpSumKernel::validate(const ITensorInfo *input, const ITensorInfo *max, const ITensorInfo *output, const ITensorInfo *sum, float beta)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_logits_1d_shift_exp_sum(input, max, output, sum, beta));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_logits_1d_shift_exp_sum(input->clone().get(), max->clone().get(), output->clone().get(), sum->clone().get()).first);

    return Status{};
}

void NELogits1DShiftExpSumKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (*_func)(_input, _max, _output, _sum, window, _beta);
}

namespace
{
void logits_1d_norm_qs8(const ITensor *in, const ITensor *sum, ITensor *out, const Window &window)
{
    Window window_sum(window);
    window_sum.set(Window::DimX, Window::Dimension(0, 0, 0));
    Window sum_slice = window_sum.first_slice_window_1D();
    Window in_slice  = window.first_slice_window_1D();

    const int fixed_point_position = in->info()->fixed_point_position();

    do
    {
        Iterator input(in, in_slice);
        Iterator _sum(sum, sum_slice);
        Iterator output(out, in_slice);

        const int8_t     sum_value        = *reinterpret_cast<const qint8_t *>(_sum.ptr());
        const qint8x16_t vec_sum_inversed = vqrecipq_qs8(vdupq_n_qs8(sum_value), fixed_point_position);

        execute_window_loop(in_slice, [&](const Coordinates & id)
        {
            const auto in_ptr  = reinterpret_cast<const qint8_t *>(input.ptr());
            const auto out_ptr = reinterpret_cast<qint8_t *>(output.ptr());

            const qint8x16_t vec_in           = vld1q_qs8(in_ptr);
            const qint8x16_t normalized_value = vqmulq_qs8(vec_in, vec_sum_inversed, fixed_point_position);

            vst1q_qs8(out_ptr, normalized_value);
        },
        input, output);
    }
    while(window.slide_window_slice_1D(in_slice) && window.slide_window_slice_1D(sum_slice));
}
void logits_1d_norm_qs16(const ITensor *in, const ITensor *sum, ITensor *out, const Window &window)
{
    Window window_sum(window);
    window_sum.set(Window::DimX, Window::Dimension(0, 0, 0));
    Window sum_slice = window_sum.first_slice_window_1D();
    Window in_slice  = window.first_slice_window_1D();

    const int fixed_point_position = in->info()->fixed_point_position();

    do
    {
        Iterator input(in, in_slice);
        Iterator _sum(sum, sum_slice);
        Iterator output(out, in_slice);

        const int16_t    sum_value        = *reinterpret_cast<const qint16_t *>(_sum.ptr());
        const qint16x8_t vec_sum_inversed = vqrecipq_qs16(vdupq_n_qs16(sum_value), fixed_point_position);

        execute_window_loop(in_slice, [&](const Coordinates & id)
        {
            const auto in_ptr  = reinterpret_cast<const qint16_t *>(input.ptr());
            const auto out_ptr = reinterpret_cast<qint16_t *>(output.ptr());

            const qint16x8_t vec_in           = vld1q_qs16(in_ptr);
            const qint16x8_t normalized_value = vqmulq_qs16(vec_in, vec_sum_inversed, fixed_point_position);

            vst1q_qs16(out_ptr, normalized_value);
        },
        input, output);
    }
    while(window.slide_window_slice_1D(in_slice) && window.slide_window_slice_1D(sum_slice));
}
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
void logits_1d_norm_f16(const ITensor *in, const ITensor *sum, ITensor *out, const Window &window)
{
    Window window_sum(window);
    window_sum.set(Window::DimX, Window::Dimension(0, 0, 0));
    Window sum_slice = window_sum.first_slice_window_1D();
    Window in_slice  = window.first_slice_window_1D();

    do
    {
        Iterator input(in, in_slice);
        Iterator _sum(sum, sum_slice);
        Iterator output(out, in_slice);

        const float16_t   sum_value        = *reinterpret_cast<const qint16_t *>(_sum.ptr());
        const float16x8_t vec_sum_inversed = vdupq_n_f16(1.0f / sum_value);

        execute_window_loop(in_slice, [&](const Coordinates & id)
        {
            const auto in_ptr  = reinterpret_cast<const float16_t *>(input.ptr());
            const auto out_ptr = reinterpret_cast<float16_t *>(output.ptr());

            const float16x8_t vec_in           = vld1q_f16(in_ptr);
            const float16x8_t normalized_value = vmulq_f16(vec_in, vec_sum_inversed);

            vst1q_f16(out_ptr, normalized_value);
        },
        input, output);
    }
    while(window.slide_window_slice_1D(in_slice) && window.slide_window_slice_1D(sum_slice));
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

void logits_1d_norm_f32(const ITensor *in, const ITensor *sum, ITensor *out, const Window &window)
{
    Window window_sum(window);
    window_sum.set(Window::DimX, Window::Dimension(0, 0, 0));
    Window sum_slice = window_sum.first_slice_window_1D();
    Window in_slice  = window.first_slice_window_1D();

    do
    {
        Iterator input(in, in_slice);
        Iterator _sum(sum, sum_slice);
        Iterator output(out, in_slice);

        const float       sum_value        = *reinterpret_cast<const float *>(_sum.ptr());
        const float32x4_t vec_sum_inversed = vdupq_n_f32(1.0f / sum_value);

        execute_window_loop(in_slice, [&](const Coordinates & id)
        {
            const auto in_ptr  = reinterpret_cast<const float *>(input.ptr());
            const auto out_ptr = reinterpret_cast<float *>(output.ptr());

            const float32x4_t vec_in           = vld1q_f32(in_ptr);
            const float32x4_t normalized_value = vmulq_f32(vec_in, vec_sum_inversed);

            vst1q_f32(out_ptr, normalized_value);
        },
        input, output);
    }
    while(window.slide_window_slice_1D(in_slice) && window.slide_window_slice_1D(sum_slice));
}
} // namespace

NELogits1DNormKernel::NELogits1DNormKernel()
    : _func(nullptr), _input(nullptr), _sum(nullptr), _output(nullptr)
{
}

void NELogits1DNormKernel::configure(const ITensor *input, const ITensor *sum, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, sum, output);

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->tensor_shape(), 1, input->info()->data_type(), input->info()->fixed_point_position());

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_logits_1d_norm(input->info(), sum->info(), output->info()));

    _input  = input;
    _sum    = sum;
    _output = output;

    switch(input->info()->data_type())
    {
        case DataType::QS8:
            _func = &logits_1d_norm_qs8;
            break;
        case DataType::QS16:
            _func = &logits_1d_norm_qs16;
            break;
        case DataType::F32:
            _func = &logits_1d_norm_f32;
            break;
        case DataType::F16:
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            _func = &logits_1d_norm_f16;
            break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        default:
            ARM_COMPUTE_ERROR("Unsupported data type.");
            break;
    }

    // Configure kernel window
    auto win_config = validate_and_configure_window_logits_1d_norm(input->info(), sum->info(), output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NELogits1DNormKernel::validate(const ITensorInfo *input, const ITensorInfo *sum, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_logits_1d_norm(input, sum, output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_logits_1d_norm(input->clone().get(), sum->clone().get(), output->clone().get()).first);

    return Status{};
}

void NELogits1DNormKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (*_func)(_input, _sum, _output, window);
}
