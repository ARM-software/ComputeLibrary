/*
 * Copyright (c) 2020-2021, 2023 Arm Limited.
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
#include "src/core/NEON/kernels/NEQLSTMLayerNormalizationKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/core/NEON/kernels/detail/NEActivationFunctionDetail.h"
#include "src/core/NEON/NEFixedPoint.h"
#include "src/core/NEON/NEMath.h"
#include "src/core/NEON/NESymm.h"

#include <map>

namespace arm_compute
{
namespace
{
inline std::pair<int64_t, int64_t> compute_mean_variance(int64_t sum, int64_t sum_sq, uint32_t num_input)
{
    const auto    temp     = static_cast<int64_t>(0x100000) / num_input;
    const auto    mean     = sum * 1024 / static_cast<int64_t>(num_input);
    const int64_t variance = ((sum_sq * temp) - (mean * mean)) / 0x100000;

    return std::make_pair(mean, variance);
}

inline int64x2x2_t mul_add(const int32x4_t &a, const int32x4_t &b, const int32x4_t &bias)
{
    using namespace wrapper;
    const int64x2_t a_low  = vmovl(vgetlow(a));
    const int64x2_t a_high = vmovl(vgethigh(a));
    const int64x2_t b_low  = vmovl(vgetlow(b));
    const int64x2_t b_high = vmovl(vgethigh(b));

    const int64_t a_0 = vgetlane(a_low, 0);
    const int64_t a_1 = vgetlane(a_low, 1);
    const int64_t a_2 = vgetlane(a_high, 0);
    const int64_t a_3 = vgetlane(a_high, 1);

    const int64_t b_0 = vgetlane(b_low, 0);
    const int64_t b_1 = vgetlane(b_low, 1);
    const int64_t b_2 = vgetlane(b_high, 0);
    const int64_t b_3 = vgetlane(b_high, 1);

    int64x2x2_t     result;
    const int64x2_t result_0{a_0 * b_0, a_1 * b_1};
    const int64x2_t result_1{a_2 * b_2, a_3 * b_3};
    result.val[0] = vadd(vmovl(vgetlow(bias)), result_0);
    result.val[1] = vadd(vmovl(vgethigh(bias)), result_1);

    return result;
}
} // namespace

void NEQLSTMLayerNormalizationKernel::configure(const ITensor *input,
                                                ITensor       *output,
                                                const ITensor *weight,
                                                const ITensor *bias)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weight, bias, output);
    ARM_COMPUTE_ERROR_ON(input == output);
    ARM_COMPUTE_ERROR_THROW_ON(validate(input->info(), output->info(), weight->info(), bias->info()));

    static const std::map<DataType, ComputeFuncType> fn_map = {
        {DataType::QSYMM16, std::mem_fn(&NEQLSTMLayerNormalizationKernel::compute_qsymm16)},
    };

    _input  = input;
    _output = output;
    _weight = weight;
    _bias   = bias;
    _fn     = fn_map.at(_input->info()->data_type());

    auto_init_if_empty(*_output->info(), *_input->info());
    _output->info()->set_quantization_info(compute_output_qinfo());

    const UniformQuantizationInfo wq_info = _weight->info()->quantization_info().uniform();
    const Status s = quantization::calculate_quantized_multiplier(wq_info.scale, &_output_multiplier, &_output_shift);
    _output_shift *= -1;

    if (!bool(s))
    {
        _output_multiplier = 0;
        _output_shift      = 0;
    }

    Window win = configure_window(output);
    INEKernel::configure(win);
}

Window NEQLSTMLayerNormalizationKernel::configure_window(ITensor *target)
{
    Window window = calculate_max_window(*target->info(), Steps());

    _window_start_x = static_cast<int32_t>(window.x().start());
    _window_end_x   = static_cast<int32_t>(window.x().end());
    _window_step_x  = static_cast<int32_t>(vector_size_byte) / _output->info()->element_size();

    // input and output windows will iterator over y-axis, while execute_window will handler x-axis.
    _inout_window = window;
    _inout_window.set(Window::DimX, Window::Dimension(0, 1, 1));

    // weight and bias cannot iterator along y-axis since they are 1D.
    _weight_window = _inout_window;
    _weight_window.set(Window::DimY, Window::Dimension(0, 1, 1));

    return window;
}

Status NEQLSTMLayerNormalizationKernel::validate(const ITensorInfo *input,
                                                 const ITensorInfo *output,
                                                 const ITensorInfo *weight,
                                                 const ITensorInfo *bias)
{
    ARM_COMPUTE_UNUSED(output, bias, weight, input);

    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weight, bias, output);

    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QSYMM16);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weight, 1, DataType::QSYMM16);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(bias, 1, DataType::S32);

    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > max_input_dimension);
    ARM_COMPUTE_RETURN_ERROR_ON(weight->num_dimensions() > max_weight_dimension);
    ARM_COMPUTE_RETURN_ERROR_ON(bias->num_dimensions() > max_bias_dimension);

    ARM_COMPUTE_RETURN_ERROR_ON(input->tensor_shape().x() != weight->tensor_shape().x());
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(weight, bias);

    if (output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
    }

    return Status{};
}

void NEQLSTMLayerNormalizationKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(window, info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON_MSG(!_fn, "internal function is not defined for computation");

    _fn(*this);
}

inline QuantizationInfo NEQLSTMLayerNormalizationKernel::compute_output_qinfo()
{
    return QuantizationInfo(1.f / 4096);
}

inline std::pair<int64_t, int64_t> NEQLSTMLayerNormalizationKernel::sum_qsymm16(const int16_t *input_ptr)
{
    ARM_COMPUTE_ERROR_ON(!input_ptr);

    using AccType       = int64_t;
    using InputDataType = int16_t;

    AccType sum{0};
    AccType sum_sq{0};

    int32_t x = _window_start_x;
    for (; x <= _window_end_x && _window_step_x <= (_window_end_x - x); x += _window_step_x)
    {
        using namespace wrapper;
        const int16x8_t val      = vloadq(input_ptr + x);
        const int32x4_t val_low  = vmovl(vgetlow(val));
        const int32x4_t val_high = vmovl(vgethigh(val));

#if defined(__aarch64__)
        sum += static_cast<AccType>(vaddv(val_low));
        sum += static_cast<AccType>(vaddv(val_high));

        sum_sq += static_cast<AccType>(vaddv(vmul(val_low, val_low)));
        sum_sq += static_cast<AccType>(vaddv(vmul(val_high, val_high)));
#else  // __aarch64__

        // only AArch64 supports vaddv
        const int64x2_t pair_sum_low  = vpaddl(val_low);
        const int64x2_t pair_sum_high = vpaddl(val_high);
        const int64x2_t pair_sum      = vadd(pair_sum_low, pair_sum_high);
        sum += vgetlane(pair_sum, 0) + vgetlane(pair_sum, 1);

        const int32x4_t square_low       = vmul(val_low, val_low);
        const int32x4_t square_high      = vmul(val_high, val_high);
        const int64x2_t pair_sum_sq_low  = vpaddl(square_low);
        const int64x2_t pair_sum_sq_high = vpaddl(square_high);
        const int64x2_t pair_sum_sq      = vadd(pair_sum_sq_low, pair_sum_sq_high);
        sum_sq += vgetlane(pair_sum_sq, 0) + vgetlane(pair_sum_sq, 1);
#endif // __aarch64__
    }

    for (; x < _window_end_x; ++x)
    {
        const InputDataType val = input_ptr[x];
        sum += static_cast<AccType>(val);
        sum_sq += static_cast<AccType>(val * val);
    }

    return std::make_pair(sum, sum_sq);
}

inline void NEQLSTMLayerNormalizationKernel::normalize_qasymm16(const int16_t *input_ptr,
                                                                int16_t       *output_ptr,
                                                                const int16_t *weight_ptr,
                                                                const int32_t *bias_ptr,
                                                                int32_t        mean,
                                                                int32_t        inv_std_mul,
                                                                int32_t        inv_std_shift)
{
    using OutputDataType = int16_t;

    using namespace wrapper;
    const int32x4_t mean_vec = vdup_n(mean, wrapper::traits::vector_128_tag{});

    int32_t x = _window_start_x;
    for (; x <= _window_end_x && _window_step_x <= (_window_end_x - x); x += _window_step_x)
    {
        const int16x8_t val = vloadq(input_ptr + x);
        int32x4x2_t     shifted;
        shifted.val[0] = vsub(vshlq_n_s32(vmovl(vgetlow(val)), 10), mean_vec);
        shifted.val[1] = vsub(vshlq_n_s32(vmovl(vgethigh(val)), 10), mean_vec);

        int32x4x2_t rescaled = multiply_by_quantized_multiplier_2row(shifted, inv_std_mul, inv_std_shift);

        const int16x8_t weight_val  = vloadq(weight_ptr + x);
        const int32x4_t weight_low  = vmovl(vgetlow(weight_val));
        const int32x4_t weight_high = vmovl(vgethigh(weight_val));

        const int32x4_t bias_low  = vloadq(bias_ptr + x);
        const int32x4_t bias_high = vloadq(bias_ptr + 4 + x);

        int64x2x2_t result_0 = mul_add(rescaled.val[0], weight_low, bias_low);
        int64x2x2_t result_1 = mul_add(rescaled.val[1], weight_high, bias_high);

        int32x4x2_t combined;
        combined.val[0] = vcombine(vmovn(vrshrq_n_s64(result_0.val[0], 10)), vmovn(vrshrq_n_s64(result_0.val[1], 10)));
        combined.val[1] = vcombine(vmovn(vrshrq_n_s64(result_1.val[0], 10)), vmovn(vrshrq_n_s64(result_1.val[1], 10)));

        int32x4x2_t out_val = multiply_by_quantized_multiplier_2row(combined, _output_multiplier, _output_shift + 12);

        vstore(output_ptr + x, vqmovn(out_val.val[0]));
        vstore(output_ptr + x + 4, vqmovn(out_val.val[1]));
    }

    for (; x < _window_end_x; ++x)
    {
        const auto    val      = static_cast<int32_t>(input_ptr[x]);
        const int32_t shifted  = (val << 10) - mean;
        const int32_t rescaled = quantization::multiply_by_quantized_multiplier(shifted, inv_std_mul, inv_std_shift);
        const int64_t weighted = rescaled * weight_ptr[x] + bias_ptr[x];
        const auto    reverse_shifted = static_cast<int32_t>((weighted + 512) >> 10);
        int32_t       out_val =
            quantization::multiply_by_quantized_multiplier(reverse_shifted, _output_multiplier, _output_shift + 12);
        out_val =
            utility::clamp<decltype(out_val), OutputDataType>(out_val, std::numeric_limits<OutputDataType>::min());
        output_ptr[x] = static_cast<OutputDataType>(out_val);
    }
}

void NEQLSTMLayerNormalizationKernel::compute_qsymm16()
{
    using InputDataType  = int16_t;
    using OutputDataType = int16_t;
    using BiasDataType   = int32_t;
    using AccType        = int64_t;

    Iterator input_iterator{_input, _inout_window};
    Iterator output_iterator{_output, _inout_window};
    Iterator weight_iterator{_weight, _weight_window};
    Iterator bias_iterator{_bias, _weight_window};

    const auto weight_ptr = reinterpret_cast<const InputDataType *>(weight_iterator.ptr());
    const auto bias_ptr   = reinterpret_cast<const BiasDataType *>(bias_iterator.ptr());

    const uint32_t column_size = _input->info()->tensor_shape()[0];

    execute_window_loop(
        _inout_window,
        [&, this](const Coordinates &)
        {
            const auto in_ptr  = reinterpret_cast<const InputDataType *>(input_iterator.ptr());
            auto       out_ptr = reinterpret_cast<OutputDataType *>(output_iterator.ptr());

            AccType sum{0};
            AccType sum_sq{0};
            std::tie(sum, sum_sq) = sum_qsymm16(in_ptr);

            AccType mean{0};
            AccType variance{0};
            std::tie(mean, variance) = compute_mean_variance(sum, sum_sq, column_size);

            int32_t stddev_invsqrt_mul{};
            int32_t stddev_invsqrt_shift{};
            quantization::get_invsqrt_quantized_multiplier_exp(static_cast<int32_t>(variance), -1, stddev_invsqrt_mul,
                                                               stddev_invsqrt_shift);

            normalize_qasymm16(in_ptr, out_ptr, weight_ptr, bias_ptr, mean, stddev_invsqrt_mul, stddev_invsqrt_shift);
        },
        input_iterator, output_iterator);
}
} // namespace arm_compute
