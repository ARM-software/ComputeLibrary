/*
 * Copyright (c) 2017-2022 Arm Limited.
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
#include "src/cpu/kernels/CpuQuantizeKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "src/core/NEON/NEAsymm.h"
#include "src/core/NEON/NEMath.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include "src/core/CPP/Validate.h"

#include <arm_neon.h>
#include <map>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
constexpr auto window_step = 16;

Status validate_arguments(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(dst->tensor_shape().total_size() == 0);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, DataType::QSYMM8, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::QASYMM16);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, dst);

    return Status{};
}

template <typename T>
inline float32x4x4_t load_value(const T *input_ptr)
{
    using Tx16_t = typename wrapper::traits::neon_vector<T, 16>::type;
    return arm_compute::convert_to_float32x4x4<Tx16_t>(wrapper::vloadq(input_ptr));
}

template <>
inline float32x4x4_t load_value(const float *input_ptr)
{
    return { wrapper::vloadq(input_ptr),
             wrapper::vloadq(input_ptr + 4),
             wrapper::vloadq(input_ptr + 8),
             wrapper::vloadq(input_ptr + 12) };
}
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <>
inline float32x4x4_t load_value(const float16_t *input_ptr)
{
    return { vcvt_f32_f16(wrapper::vload(input_ptr)),
             vcvt_f32_f16(wrapper::vload(input_ptr + 4)),
             vcvt_f32_f16(wrapper::vload(input_ptr + 8)),
             vcvt_f32_f16(wrapper::vload(input_ptr + 12)) };
}

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

template <typename element_type>
using vector_type = wrapper::traits::neon_vector_t<element_type, window_step>;

template <typename quantized_type>
vector_type<quantized_type> vquantize_qasymm8(const float32x4x4_t &qv, const UniformQuantizationInfo &qi);

template <>
vector_type<uint8_t> vquantize_qasymm8<uint8_t>(const float32x4x4_t &qv, const UniformQuantizationInfo &qi)
{
    return vquantize(qv, qi);
}

template <>
vector_type<int8_t> vquantize_qasymm8<int8_t>(const float32x4x4_t &qv, const UniformQuantizationInfo &qi)
{
    return vquantize_signed(qv, qi);
}

} // namespace

void CpuQuantizeKernel::configure(const ITensorInfo *src, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst));

    static const std::map<std::string, QuantizeFunctionExecutorPtr> quant_map =
    {
        { "op_QASYMM8_QASYMM8", &CpuQuantizeKernel::run_quantize_qasymm8<uint8_t, uint8_t> },
        { "op_QASYMM8_QASYMM8_SIGNED", &CpuQuantizeKernel::run_quantize_qasymm8<uint8_t, int8_t> },
        { "op_QASYMM8_QASYMM16", &CpuQuantizeKernel::run_quantize_qasymm16<uint8_t> },

        { "op_QASYMM8_SIGNED_QASYMM8", &CpuQuantizeKernel::run_quantize_qasymm8<int8_t, uint8_t> },
        { "op_QASYMM8_SIGNED_QASYMM8_SIGNED", &CpuQuantizeKernel::run_quantize_qasymm8<int8_t, int8_t> },
        { "op_QASYMM8_SIGNED_QASYMM16", &CpuQuantizeKernel::run_quantize_qasymm16<int8_t> },

        { "op_F32_QSYMM8", &CpuQuantizeKernel::run_quantize_qsymm8<float, int8_t> },

        { "op_F32_QASYMM8", &CpuQuantizeKernel::run_quantize_qasymm8<float, uint8_t> },
        { "op_F32_QASYMM8_SIGNED", &CpuQuantizeKernel::run_quantize_qasymm8<float, int8_t> },
        { "op_F32_QASYMM16", &CpuQuantizeKernel::run_quantize_qasymm16<float> },

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        { "op_F16_QASYMM8", &CpuQuantizeKernel::run_quantize_qasymm8<float16_t, uint8_t> },
        { "op_F16_QASYMM8_SIGNED", &CpuQuantizeKernel::run_quantize_qasymm8<float16_t, int8_t> },
        { "op_F16_QASYMM16", &CpuQuantizeKernel::run_quantize_qasymm16<float16_t> },
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC*/
    };

    std::string function_to_call("op_");
    function_to_call += string_from_data_type(src->data_type()) + "_";
    function_to_call += string_from_data_type(dst->data_type());

    auto it = quant_map.find(function_to_call);

    if(it == quant_map.end())
    {
        ARM_COMPUTE_ERROR("Unsupported combination of input and output data types");
    }
    _func = it->second;

    // Configure kernel window
    Window win_config = calculate_max_window(*src, Steps());
    ICpuKernel::configure(win_config);
}

Status CpuQuantizeKernel::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst));
    return Status{};
}

template <typename TIn, typename TOut>
void CpuQuantizeKernel::run_quantize_qsymm8(const ITensor *src, ITensor *dst, const Window &window)
{
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    const UniformQuantizationInfo uqinfo_in = src->info()->quantization_info().uniform();
    UniformQuantizationInfo       uqinfo    = dst->info()->quantization_info().uniform();
    if(is_data_type_quantized_asymmetric(src->info()->data_type()))
    {
        uqinfo = compute_requantization_scale_offset(uqinfo_in, uqinfo);
    }
    // Collapse window and reset first dimension to handle tail calculations manually
    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(src, win_collapsed);
    Iterator output(dst, win_collapsed);
    execute_window_loop(win_collapsed, [&](const Coordinates &)
    {
        auto input_ptr  = reinterpret_cast<const TIn *>(input.ptr());
        auto output_ptr = reinterpret_cast<TOut *>(output.ptr());
        int x = window_start_x;
        for(; x <= (window_end_x - window_step); x += window_step)
        {
            wrapper::vstore(&output_ptr[x], vquantize_qasymm8<TOut>(load_value(&input_ptr[x]), uqinfo));
        }
        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            output_ptr[x] = quantize_qsymm8(input_ptr[x], dst->info()->quantization_info());
        }
    },
    input, output);
}

template <typename TIn, typename TOut>
void CpuQuantizeKernel::run_quantize_qasymm8(const ITensor *src, ITensor *dst, const Window &window)
{
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    const UniformQuantizationInfo uqinfo_in = src->info()->quantization_info().uniform();
    UniformQuantizationInfo       uqinfo    = dst->info()->quantization_info().uniform();
    if(is_data_type_quantized_asymmetric(src->info()->data_type()))
    {
        uqinfo = compute_requantization_scale_offset(uqinfo_in, uqinfo);
    }
#ifdef __aarch64__
    constexpr RoundingPolicy rounding_policy = RoundingPolicy::TO_NEAREST_EVEN;
#else  //__aarch64__
    constexpr RoundingPolicy rounding_policy = RoundingPolicy::TO_ZERO;
#endif //__aarch64__

    // Collapse window and reset first dimension to handle tail calculations manually
    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(src, win_collapsed);
    Iterator output(dst, win_collapsed);
    execute_window_loop(win_collapsed, [&](const Coordinates &)
    {
        auto input_ptr  = reinterpret_cast<const TIn *>(input.ptr());
        auto output_ptr = reinterpret_cast<TOut *>(output.ptr());

        int x = window_start_x;
        for(; x <= (window_end_x - window_step); x += window_step)
        {
            wrapper::vstore(&output_ptr[x], vquantize_qasymm8<TOut>(load_value(&input_ptr[x]), uqinfo));
        }
        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            output_ptr[x] = Qasymm8QuantizationHelper<TOut>::quantize(input_ptr[x], uqinfo, rounding_policy);
        }
    },
    input, output);
}

template <typename T>
void CpuQuantizeKernel::run_quantize_qasymm16(const ITensor *src, ITensor *dst, const Window &window)
{
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    const UniformQuantizationInfo uqinfo_in = src->info()->quantization_info().uniform();
    UniformQuantizationInfo       uqinfo    = dst->info()->quantization_info().uniform();
    if(is_data_type_quantized_asymmetric(src->info()->data_type()))
    {
        uqinfo = compute_requantization_scale_offset(uqinfo_in, uqinfo);
    }
#ifdef __aarch64__
    constexpr RoundingPolicy rounding_policy = RoundingPolicy::TO_NEAREST_EVEN;
#else  //__aarch64__
    constexpr RoundingPolicy rounding_policy = RoundingPolicy::TO_ZERO;
#endif //__aarch64__

    // Collapse window and reset first dimension to handle tail calculations manually
    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(src, win_collapsed);
    Iterator output(dst, win_collapsed);
    execute_window_loop(win_collapsed, [&](const Coordinates &)
    {
        auto input_ptr  = reinterpret_cast<const T *>(input.ptr());
        auto output_ptr = reinterpret_cast<uint16_t *>(output.ptr());

        int x = window_start_x;
        for(; x <= (window_end_x - window_step); x += window_step)
        {
            uint16x8x2_t tmp = vquantize_qasymm16(load_value(&input_ptr[x]), uqinfo);
            vst1q_u16(&output_ptr[x], tmp.val[0]);
            vst1q_u16(&output_ptr[x + 8], tmp.val[1]);
        }
        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            output_ptr[x] = quantize_qasymm16(input_ptr[x], uqinfo, rounding_policy);
        }
    },
    input, output);
}

void CpuQuantizeKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    const auto src = tensors.get_const_tensor(TensorType::ACL_SRC);
    auto       dst = tensors.get_tensor(TensorType::ACL_DST);
    (this->*_func)(src, dst, window);
}

const char *CpuQuantizeKernel::name() const
{
    return "CpuQuantizeKernel";
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
