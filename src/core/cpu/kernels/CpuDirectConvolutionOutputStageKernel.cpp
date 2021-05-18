/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "src/core/cpu/kernels/CpuDirectConvolutionOutputStageKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/Traits.h"
#include "src/core/CPP/Validate.h"
#include "src/core/NEON/NEAsymm.h"
#include "src/core/NEON/NEFixedPoint.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
Status validate_arguments(const ITensorInfo *src, const ITensorInfo *bias, const ITensorInfo *dst,
                          const DirectConvolutionLayerOutputStageKernelInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON(src->data_layout() == DataLayout::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::F16, DataType::S32, DataType::F32);

    if(bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, bias);
        ARM_COMPUTE_RETURN_ERROR_ON(bias->dimension(0) != src->dimension(get_data_layout_dimension_index(src->data_layout(), DataLayoutDimension::CHANNEL)));
        ARM_COMPUTE_RETURN_ERROR_ON(bias->num_dimensions() > 1);
    }

    if(src->data_type() == DataType::S32)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(dst == nullptr, "In-place computation not allowed for quantized output");
    }

    // Checks performed when output is configured
    if((dst != nullptr) && (dst->total_size() != 0))
    {
        if(is_data_type_float(src->data_type()))
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED);
        }
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, dst);
    }
    else if(src->data_type() == DataType::S32)
    {
        // In case of quantized computation and unconfigured output, the output data type must be provided through DirectConvolutionLayerOutputStageKernelInfo
        ARM_COMPUTE_RETURN_ERROR_ON((info.output_data_type != DataType::QASYMM8) && (info.output_data_type != DataType::QASYMM8_SIGNED));
    }

    return Status{};
}

template <typename T>
typename std::enable_if<arm_compute::utils::traits::is_floating_point<T>::value, void>::type
output_stage_nchw(ITensor *src, const ITensor *bias, const Window &window, ITensor *dst,
                  int result_fixedpoint_multiplier, int result_shift, int result_offset_after_shift)
{
    const bool has_bias = bias != nullptr;
    /** SIMD vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_bitvector_tag_t<T, wrapper::traits::BitWidth::W128>;

    ARM_COMPUTE_ERROR_ON(src->info()->data_layout() == DataLayout::UNKNOWN);
    ARM_COMPUTE_UNUSED(result_fixedpoint_multiplier);
    ARM_COMPUTE_UNUSED(result_shift);
    ARM_COMPUTE_UNUSED(result_offset_after_shift);

    const int window_start_x = window.x().start();
    const int window_end_x   = window.x().end();
    const int window_step_x  = 16 / src->info()->element_size();
    Window    win            = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator in(src, win);
    Iterator out(dst, win);
    execute_window_loop(win, [&](const Coordinates & id)
    {
        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            // Get bias and pointer to input
            const auto in_ptr = reinterpret_cast<const T *>(in.ptr()) + x;
            auto       v_in   = wrapper::vloadq(in_ptr);

            // Accumulate bias
            if(has_bias)
            {
                const auto vb = wrapper::vdup_n(*reinterpret_cast<const T *>(bias->ptr_to_element(Coordinates(id.z()))), ExactTagType{});
                v_in          = wrapper::vadd(v_in, vb);
            }

            const auto out_ptr = reinterpret_cast<T *>(out.ptr()) + x;
            wrapper::vstore(out_ptr, v_in);
        }

        // Left-overs loop
        for(; x < window_end_x; ++x)
        {
            // Get bias and pointer to input
            auto s_in = *(reinterpret_cast<const T *>(in.ptr()) + x);

            // Accumulate bias
            if(has_bias)
            {
                const auto b = *reinterpret_cast<const T *>(bias->ptr_to_element(Coordinates(id.z())));
                s_in += b;
            }

            *(reinterpret_cast<T *>(out.ptr()) + x) = s_in;
        }

    },
    in, out);
}

template <typename T>
typename std::enable_if<arm_compute::utils::traits::is_floating_point<T>::value, void>::type
output_stage_nhwc(ITensor *src, const ITensor *bias, const Window &window, ITensor *dst,
                  int result_fixedpoint_multiplier, int result_shift, int result_offset_after_shift)
{
    const bool has_bias = bias != nullptr;
    ARM_COMPUTE_UNUSED(result_fixedpoint_multiplier);
    ARM_COMPUTE_UNUSED(result_shift);
    ARM_COMPUTE_UNUSED(result_offset_after_shift);

    Window window_bias = window;
    window_bias.set(Window::DimX, Window::Dimension(0, 1, 1));
    window_bias.set(Window::DimY, Window::Dimension(0, 0, 0));
    window_bias.set(Window::DimZ, Window::Dimension(0, 0, 0));
    window_bias.set(3, Window::Dimension(0, 0, 0));

    const int window_start_x = window.x().start();
    const int window_end_x   = window.x().end();
    const int window_step_x  = 16 / src->info()->element_size();
    Window    win            = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator in(src, win);
    Iterator bi(bias, window_bias);
    Iterator out(dst, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            // Get bias and pointer to input
            const auto in_ptr = reinterpret_cast<const T *>(in.ptr());
            auto       v_in   = wrapper::vloadq(in_ptr + x);

            // Accumulate bias
            if(has_bias)
            {
                const auto bias_ptr = reinterpret_cast<T *>(bi.ptr()) + x;
                v_in                = wrapper::vadd(v_in, wrapper::vloadq(bias_ptr));
            }

            const auto out_ptr = reinterpret_cast<T *>(out.ptr());
            wrapper::vstore(out_ptr + x, v_in);
        }

        // Left-overs loop
        for(; x < window_end_x; ++x)
        {
            // Get bias and pointer to input
            auto s_in = *(reinterpret_cast<const T *>(in.ptr()) + x);

            // Accumulate bias
            if(has_bias)
            {
                const auto bias_ptr = reinterpret_cast<T *>(bi.ptr()) + x;
                s_in += *bias_ptr;
            }

            const auto out_ptr = reinterpret_cast<T *>(out.ptr());
            *(out_ptr + x)     = s_in;
        }
    },
    in, bi, out);
}

// Quantized case
template < typename TOut, typename std::enable_if < std::is_same<TOut, uint8_t>::value || std::is_same<TOut, int8_t>::value, int >::type = 0 >
void output_stage_nchw(ITensor *src, const ITensor *bias, const Window &window, ITensor *dst,
                       int result_fixedpoint_multiplier, int result_shift, int result_offset_after_shift)
{
    const bool has_bias = bias != nullptr;
    using VectorType    = typename wrapper::traits::neon_bitvector_t<TOut, wrapper::traits::BitWidth::W128>;
    using TagType       = typename wrapper::traits::neon_bitvector_tag_t<TOut, wrapper::traits::BitWidth::W128>;

    const int32x4_t result_offset_after_shift_s32 = vdupq_n_s32(result_offset_after_shift);

    const VectorType min = wrapper::vdup_n(std::numeric_limits<TOut>::lowest(), TagType{});
    const VectorType max = wrapper::vdup_n(std::numeric_limits<TOut>::max(), TagType{});

    const int window_start_x = window.x().start();
    const int window_end_x   = window.x().end();
    const int window_step_x  = 16 / src->info()->element_size();
    Window    win            = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator in(src, win);
    Iterator out(dst, win);

    execute_window_loop(win, [&](const Coordinates & id)
    {

        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            // Get bias and pointer to input
            const auto  in_ptr = reinterpret_cast<int32_t *>(in.ptr()) + x;
            int32x4x4_t v_in =
            {
                {
                    wrapper::vloadq(in_ptr),
                    wrapper::vloadq(in_ptr + 4),
                    wrapper::vloadq(in_ptr + 8),
                    wrapper::vloadq(in_ptr + 12)
                }
            };

            // Accumulate bias
            if(has_bias)
            {
                const auto vb = wrapper::vdup_n(*reinterpret_cast<const int32_t *>(bias->ptr_to_element(Coordinates(id.z()))), TagType{});
                v_in =
                {
                    {
                        wrapper::vadd(v_in.val[0], vb),
                        wrapper::vadd(v_in.val[1], vb),
                        wrapper::vadd(v_in.val[2], vb),
                        wrapper::vadd(v_in.val[3], vb)
                    }
                };
            }

            const auto out_ptr = reinterpret_cast<TOut *>(out.ptr()) + x;
            wrapper::vstore(out_ptr, finalize_quantization(v_in, result_fixedpoint_multiplier, result_shift, result_offset_after_shift_s32,
                                                           min, max, false));
        }

        // Left-overs loop
        for(; x < window_end_x; ++x)
        {
            // Get bias and pointer to input
            int32_t s_in = *(reinterpret_cast<const int32_t *>(in.ptr()) + x);

            // Accumulate bias
            if(has_bias)
            {
                const auto b = *reinterpret_cast<const int32_t *>(bias->ptr_to_element(Coordinates(id.z())));
                s_in += b;
            }

            const auto out_ptr = reinterpret_cast<TOut *>(out.ptr()) + x;
            *out_ptr           = finalize_quantization(s_in, result_fixedpoint_multiplier, result_shift, result_offset_after_shift,
                                                       std::numeric_limits<TOut>::lowest(), std::numeric_limits<TOut>::max(), false);
        }
    },
    in, out);
}
template < typename TOut, typename std::enable_if < std::is_same<TOut, uint8_t>::value || std::is_same<TOut, int8_t>::value, int >::type = 0 >
void output_stage_nhwc(ITensor *src, const ITensor *bias, const Window &window, ITensor *dst,
                       int result_fixedpoint_multiplier, int result_shift, int result_offset_after_shift)
{
    const bool has_bias = bias != nullptr;
    using VectorType    = typename wrapper::traits::neon_bitvector_t<TOut, wrapper::traits::BitWidth::W128>;
    using TagType       = typename wrapper::traits::neon_bitvector_tag_t<TOut, wrapper::traits::BitWidth::W128>;

    const int32x4_t result_offset_after_shift_s32 = vdupq_n_s32(result_offset_after_shift);

    const VectorType min = wrapper::vdup_n(std::numeric_limits<TOut>::lowest(), TagType{});
    const VectorType max = wrapper::vdup_n(std::numeric_limits<TOut>::max(), TagType{});

    Window window_bias = window;
    window_bias.set(Window::DimX, Window::Dimension(0, 1, 1));
    window_bias.set(Window::DimY, Window::Dimension(0, 0, 0));
    window_bias.set(Window::DimZ, Window::Dimension(0, 0, 0));
    window_bias.set(3, Window::Dimension(0, 0, 0));

    const int window_start_x = window.x().start();
    const int window_end_x   = window.x().end();
    const int window_step_x  = 16 / src->info()->element_size();
    Window    win            = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator in(src, win);
    Iterator bi(bias, window_bias);
    Iterator out(dst, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            // Get bias and pointer to input
            const auto  in_ptr = reinterpret_cast<int32_t *>(in.ptr()) + x;
            int32x4x4_t v_in =
            {
                {
                    wrapper::vloadq(in_ptr),
                    wrapper::vloadq(in_ptr + 4),
                    wrapper::vloadq(in_ptr + 8),
                    wrapper::vloadq(in_ptr + 12),
                }
            };

            // Accumulate bias
            if(has_bias)
            {
                const auto bias_ptr = reinterpret_cast<int32_t *>(bi.ptr()) + x;

                wrapper::vadd(v_in.val[0], wrapper::vloadq(bias_ptr));
                wrapper::vadd(v_in.val[1], wrapper::vloadq(bias_ptr + 4));
                wrapper::vadd(v_in.val[2], wrapper::vloadq(bias_ptr + 8));
                wrapper::vadd(v_in.val[3], wrapper::vloadq(bias_ptr + 12));
            }

            const auto out_ptr = reinterpret_cast<TOut *>(out.ptr()) + x;
            wrapper::vstore(out_ptr, finalize_quantization(v_in, result_fixedpoint_multiplier, result_shift, result_offset_after_shift_s32, min, max, false));
        }

        // Left-overs loop
        for(; x < window_end_x; ++x)
        {
            // Get bias and pointer to input
            const auto in_ptr = reinterpret_cast<int32_t *>(in.ptr()) + x;
            int32_t    s_in   = *in_ptr;

            // Accumulate bias
            if(has_bias)
            {
                const auto bias_ptr = reinterpret_cast<int32_t *>(bi.ptr()) + x;
                s_in += *bias_ptr;
            }

            const auto out_ptr = reinterpret_cast<TOut *>(out.ptr()) + x;
            *out_ptr           = finalize_quantization(s_in, result_fixedpoint_multiplier, result_shift, result_offset_after_shift,
                                                       std::numeric_limits<TOut>::lowest(), std::numeric_limits<TOut>::max(), false);
        }
    },
    in, bi, out);
}
} // namespace

void CpuDirectConvolutionOutputStageKernel::configure(ITensorInfo *src, const ITensorInfo *bias, ITensorInfo *dst,
                                                      const DirectConvolutionLayerOutputStageKernelInfo &info)
{
    ARM_COMPUTE_UNUSED(bias);
    // Perform validation step
    ARM_COMPUTE_ERROR_ON_NULLPTR(src);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, bias, dst, info));

    _func                         = nullptr;
    _result_fixedpoint_multiplier = info.result_fixedpoint_multiplier;
    _result_shift                 = info.result_shift;
    _result_offset_after_shift    = info.result_offset_after_shift;

    // Auto-initialize output output if required
    if(dst != nullptr)
    {
        // Work out expected output data type
        const DataType output_dt = (src->data_type() == DataType::S32) ? info.output_data_type : DataType::S32;
        // Output tensor auto initialization if not yet initialized
        auto_init_if_empty(*dst, src->clone()->set_data_type(output_dt));
    }

    Window win = calculate_max_window(*src, Steps());

    ICpuKernel::configure(win);

    const bool is_qasymm8_signed = (dst != nullptr) ? is_data_type_quantized_asymmetric_signed(dst->data_type()) : false;

    // Set appropriate function
    if(src->data_layout() == DataLayout::NCHW)
    {
        switch(src->data_type())
        {
            case DataType::S32:
            {
                if(is_qasymm8_signed)
                {
                    _func = &output_stage_nchw<int8_t>;
                }
                else
                {
                    _func = &output_stage_nchw<uint8_t>;
                }
                break;
            }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            case DataType::F16:
            {
                _func = &output_stage_nchw<float16_t>;
                break;
            }
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
            case DataType::F32:
            {
                _func = &output_stage_nchw<float>;
                break;
            }
            default:
            {
                ARM_COMPUTE_ERROR("Unsupported combination of types among the inputs.");
            }
        }
    }
    else
    {
        switch(src->data_type())
        {
            case DataType::S32:
            {
                if(is_qasymm8_signed)
                {
                    _func = &output_stage_nhwc<int8_t>;
                }
                else
                {
                    _func = &output_stage_nhwc<uint8_t>;
                }
                break;
            }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            case DataType::F16:
            {
                _func = &output_stage_nhwc<float16_t>;
                break;
            }
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
            case DataType::F32:
            {
                _func = &output_stage_nhwc<float>;
                break;
            }
            default:
            {
                ARM_COMPUTE_ERROR("Unsupported combination of types among the inputs.");
            }
        }
    }
}

Status CpuDirectConvolutionOutputStageKernel::validate(const ITensorInfo *src, const ITensorInfo *bias, const ITensorInfo *dst,
                                                       const DirectConvolutionLayerOutputStageKernelInfo &info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, bias, dst, info));
    return Status{};
}

void CpuDirectConvolutionOutputStageKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    auto src  = tensors.get_tensor(TensorType::ACL_SRC_0);
    auto bias = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    auto dst  = tensors.get_tensor(TensorType::ACL_DST);

    (*_func)(src, bias, window, dst, _result_fixedpoint_multiplier, _result_shift, _result_offset_after_shift);
}

const char *CpuDirectConvolutionOutputStageKernel::name() const
{
    return "CpuDirectConvolutionOutputStageKernel";
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
