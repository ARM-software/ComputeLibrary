/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEActivationLayerKernel.h"

#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEAsymm.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/NEON/NEMath.h"
#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include "arm_compute/core/QAsymm8.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>
#include <array>
#include <cmath>
#include <map>

using namespace arm_compute;
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::QASYMM8, DataType::F16, DataType::F32);

    // Checks performed when output is configured
    if((output != nullptr) && (output->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    // Configure kernel window
    Window win = calculate_max_window(*input, Steps());

    if(output != nullptr)
    {
        // Output auto inizialitation if not yet initialized
        auto_init_if_empty(*output, *input->clone());

        // NEActivationLayerKernel doesn't need padding so update_window_and_padding() can be skipped
        Coordinates coord;
        coord.set_num_dimensions(output->num_dimensions());
        output->set_valid_region(ValidRegion(coord, output->tensor_shape()));
    }

    return std::make_pair(Status{}, win);
}
} // namespace

NEActivationLayerKernel::NEActivationLayerKernel()
    : _input(nullptr), _output(nullptr), _func(nullptr), _act_info(ActivationFunction::LOGISTIC)
{
}

void NEActivationLayerKernel::configure(ITensor *input, ITensor *output, ActivationLayerInfo activation_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input);

    _input    = input;
    _act_info = activation_info;
    _output   = input;

    if(output != nullptr)
    {
        _output = output;
    }

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), (output != nullptr) ? output->info() : nullptr));

    ARM_COMPUTE_ERROR_ON_MSG((input->info()->data_type() == DataType::QASYMM8) && (activation_info.activation() != ActivationLayerInfo::ActivationFunction::RELU)
                             && (activation_info.activation() != ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU) && (activation_info.activation() != ActivationLayerInfo::ActivationFunction::BOUNDED_RELU)
                             && (activation_info.activation() != ActivationLayerInfo::ActivationFunction::LOGISTIC),
                             "For QASYMM8 only logistic, relu and lower/upper bounded relu are supported");

    // Activation functions : FP32
    static std::map<ActivationFunction, ActivationFunctionExecutorPtr> act_map_f32 =
    {
        { ActivationFunction::ABS, &NEActivationLayerKernel::activation<ActivationFunction::ABS, float> },
        { ActivationFunction::LINEAR, &NEActivationLayerKernel::activation<ActivationFunction::LINEAR, float> },
        { ActivationFunction::LOGISTIC, &NEActivationLayerKernel::activation<ActivationFunction::LOGISTIC, float> },
        { ActivationFunction::RELU, &NEActivationLayerKernel::activation<ActivationFunction::RELU, float> },
        { ActivationFunction::BOUNDED_RELU, &NEActivationLayerKernel::activation<ActivationFunction::BOUNDED_RELU, float> },
        { ActivationFunction::LU_BOUNDED_RELU, &NEActivationLayerKernel::activation<ActivationFunction::LU_BOUNDED_RELU, float> },
        { ActivationFunction::LEAKY_RELU, &NEActivationLayerKernel::activation<ActivationFunction::LEAKY_RELU, float> },
        { ActivationFunction::SOFT_RELU, &NEActivationLayerKernel::activation<ActivationFunction::SOFT_RELU, float> },
        { ActivationFunction::SQRT, &NEActivationLayerKernel::activation<ActivationFunction::SQRT, float> },
        { ActivationFunction::SQUARE, &NEActivationLayerKernel::activation<ActivationFunction::SQUARE, float> },
        { ActivationFunction::TANH, &NEActivationLayerKernel::activation<ActivationFunction::TANH, float> },
    };

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    // Activation functions : FP16
    static std::map<ActivationFunction, ActivationFunctionExecutorPtr> act_map_f16 =
    {
        { ActivationFunction::ABS, &NEActivationLayerKernel::activation<ActivationFunction::ABS, float16_t> },
        { ActivationFunction::LINEAR, &NEActivationLayerKernel::activation<ActivationFunction::LINEAR, float16_t> },
        { ActivationFunction::LOGISTIC, &NEActivationLayerKernel::activation<ActivationFunction::LOGISTIC, float16_t> },
        { ActivationFunction::RELU, &NEActivationLayerKernel::activation<ActivationFunction::RELU, float16_t> },
        { ActivationFunction::BOUNDED_RELU, &NEActivationLayerKernel::activation<ActivationFunction::BOUNDED_RELU, float16_t> },
        { ActivationFunction::LU_BOUNDED_RELU, &NEActivationLayerKernel::activation<ActivationFunction::LU_BOUNDED_RELU, float16_t> },
        { ActivationFunction::LEAKY_RELU, &NEActivationLayerKernel::activation<ActivationFunction::LEAKY_RELU, float16_t> },
        { ActivationFunction::SOFT_RELU, &NEActivationLayerKernel::activation<ActivationFunction::SOFT_RELU, float16_t> },
        { ActivationFunction::SQRT, &NEActivationLayerKernel::activation<ActivationFunction::SQRT, float16_t> },
        { ActivationFunction::SQUARE, &NEActivationLayerKernel::activation<ActivationFunction::SQUARE, float16_t> },
        { ActivationFunction::TANH, &NEActivationLayerKernel::activation<ActivationFunction::TANH, float16_t> },
    };
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC*/

    // Activation functions : QASYMM8
    static std::map<ActivationFunction, ActivationFunctionExecutorPtr> act_map_qasymm8 =
    {
        { ActivationFunction::LOGISTIC, &NEActivationLayerKernel::activation<ActivationFunction::LOGISTIC, qasymm8_t> },
        { ActivationFunction::BOUNDED_RELU, &NEActivationLayerKernel::activation<ActivationFunction::BOUNDED_RELU, qasymm8_t> },
        { ActivationFunction::LU_BOUNDED_RELU, &NEActivationLayerKernel::activation<ActivationFunction::LU_BOUNDED_RELU, qasymm8_t> },
        { ActivationFunction::RELU, &NEActivationLayerKernel::activation<ActivationFunction::RELU, qasymm8_t> },
    };

    switch(input->info()->data_type())
    {
        case DataType::QASYMM8:
            _func = act_map_qasymm8[activation_info.activation()];
            break;
        case DataType::F32:
            _func = act_map_f32[activation_info.activation()];
            break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            _func = act_map_f16[activation_info.activation()];
            break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        default:
            ARM_COMPUTE_ERROR("Unsupported data type.");
    }

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), (output != nullptr) ? output->info() : nullptr);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICPPKernel::configure(win_config.second);
}

template <ActivationLayerInfo::ActivationFunction F, typename T>
typename std::enable_if<arm_compute::utils::traits::is_floating_point<T>::value, void>::type
NEActivationLayerKernel::activation(const Window &window)
{
    /** NEON vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_bitvector_tag_t<T, wrapper::traits::BitWidth::W128>;

    const int                window_step_x  = 16 / sizeof(T);
    const auto               window_start_x = static_cast<int>(window.x().start());
    const auto               window_end_x   = static_cast<int>(window.x().end());
    const ActivationFunction act            = F;

    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(_input, win_collapsed);
    Iterator output(_output, win_collapsed);

    const auto const_1 = wrapper::vdup_n(static_cast<T>(1.f), ExactTagType{});
    const auto const_0 = wrapper::vdup_n(static_cast<T>(0.f), ExactTagType{});
    const auto va      = wrapper::vdup_n(static_cast<T>(_act_info.a()), ExactTagType{});
    const auto vb      = wrapper::vdup_n(static_cast<T>(_act_info.b()), ExactTagType{});
    const auto a       = static_cast<T>(_act_info.a());
    const auto b       = static_cast<T>(_act_info.b());

    execute_window_loop(win_collapsed, [&](const Coordinates &)
    {
        const auto input_ptr  = reinterpret_cast<const T *>(input.ptr());
        const auto output_ptr = reinterpret_cast<T *>(output.ptr());

        wrapper::traits::neon_bitvector_t<T, wrapper::traits::BitWidth::W128> tmp;

        // Compute S elements per iteration
        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const auto vin = wrapper::vloadq(input_ptr + x);
            switch(act)
            {
                case ActivationFunction::ABS:
                    tmp = wrapper::vabs(vin);
                    break;
                case ActivationFunction::LINEAR:
                    tmp = wrapper::vmla(vb, va, vin);
                    break;
                case ActivationFunction::LOGISTIC:
                    tmp = wrapper::vinv(wrapper::vadd(const_1, wrapper::vexpq(wrapper::vneg(vin))));
                    break;
                case ActivationFunction::RELU:
                    tmp = wrapper::vmax(const_0, vin);
                    break;
                case ActivationFunction::BOUNDED_RELU:
                    tmp = wrapper::vmin(va, wrapper::vmax(const_0, vin));
                    break;
                case ActivationFunction::LU_BOUNDED_RELU:
                    tmp = wrapper::vmin(va, wrapper::vmax(vb, vin));
                    break;
                case ActivationFunction::LEAKY_RELU:
                    tmp = wrapper::vbsl(wrapper::vcgt(vin, const_0), vin, wrapper::vmul(va, vin));
                    break;
                case ActivationFunction::SOFT_RELU:
                    tmp = wrapper::vlog(wrapper::vadd(const_1, wrapper::vexpq(vin)));
                    break;
                case ActivationFunction::SQRT:
                    tmp = wrapper::vinv(wrapper::vinvsqrt(vin));
                    break;
                case ActivationFunction::SQUARE:
                    tmp = wrapper::vmul(vin, vin);
                    break;
                case ActivationFunction::TANH:
                    tmp = wrapper::vmul(va, wrapper::vtanh(wrapper::vmul(vb, vin)));
                    break;
                default:
                    ARM_COMPUTE_ERROR("Unsupported activation function");
            }
            wrapper::vstore(output_ptr + x, tmp);
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            const T in = *(reinterpret_cast<const T *>(input_ptr + x));
            T       tmp;
            switch(act)
            {
                case ActivationFunction::ABS:
                    tmp = std::abs(in);
                    break;
                case ActivationFunction::LINEAR:
                    tmp = a * in + b;
                    break;
                case ActivationFunction::LOGISTIC:
                    tmp = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-in));
                    break;
                case ActivationFunction::RELU:
                    tmp = std::max<T>(static_cast<T>(0), in);
                    break;
                case ActivationFunction::BOUNDED_RELU:
                    tmp = std::min<T>(a, std::max(static_cast<T>(0), in));
                    break;
                case ActivationFunction::LU_BOUNDED_RELU:
                    tmp = std::min<T>(a, std::max<T>(b, in));
                    break;
                case ActivationFunction::LEAKY_RELU:
                    tmp = (in > 0) ? in : a * in;
                    break;
                case ActivationFunction::SOFT_RELU:
                    tmp = std::log(static_cast<T>(1) + std::exp(in));
                    break;
                case ActivationFunction::SQRT:
                    tmp = std::sqrt(in);
                    break;
                case ActivationFunction::SQUARE:
                    tmp = in * in;
                    break;
                case ActivationFunction::TANH:
                    tmp = a * std::tanh(b * in);
                    break;
                default:
                    ARM_COMPUTE_ERROR("Unsupported activation function");
            }
            *(output_ptr + x) = tmp;
        }
    },
    input, output);
}

template <ActivationLayerInfo::ActivationFunction F, typename T>
typename std::enable_if<std::is_same<T, qasymm8_t>::value, void>::type NEActivationLayerKernel::activation(const Window &window)
{
    const int                window_step_x  = 16 / sizeof(T);
    const auto               window_start_x = static_cast<int>(window.x().start());
    const auto               window_end_x   = static_cast<int>(window.x().end());
    const ActivationFunction act            = F;

    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(_input, win_collapsed);
    Iterator output(_output, win_collapsed);

    const QuantizationInfo qi_in    = _input->info()->quantization_info();
    const QuantizationInfo qi_out   = _output->info()->quantization_info();
    const qasymm8x16_t     va       = vdupq_n_u8(sqcvt_qasymm8_f32(_act_info.a(), qi_in.scale, qi_in.offset));
    const qasymm8x16_t     vb       = vdupq_n_u8(sqcvt_qasymm8_f32(_act_info.b(), qi_in.scale, qi_in.offset));
    const qasymm8_t        a        = sqcvt_qasymm8_f32(_act_info.a(), qi_in.scale, qi_in.offset);
    const qasymm8_t        b        = sqcvt_qasymm8_f32(_act_info.b(), qi_in.scale, qi_in.offset);
    const qasymm8_t        const_0  = sqcvt_qasymm8_f32(0.f, qi_in.scale, qi_in.offset);
    const qasymm8x16_t     vconst_0 = vdupq_n_u8(const_0);
    const auto             vconst_1 = vdupq_n_f32(1.f);

    // Initialise scale/offset for re-quantization
    float       s  = qi_in.scale / qi_out.scale;
    float       o  = -qi_in.offset * s + qi_out.offset;
    float32x4_t vs = vdupq_n_f32(s);
    float32x4_t vo = vdupq_n_f32(o);

    execute_window_loop(win_collapsed, [&](const Coordinates &)
    {
        const auto input_ptr  = reinterpret_cast<const T *>(input.ptr());
        const auto output_ptr = reinterpret_cast<T *>(output.ptr());

        wrapper::traits::neon_bitvector_t<T, wrapper::traits::BitWidth::W128> tmp;

        // Compute S elements per iteration
        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const auto vin = wrapper::vloadq(input_ptr + x);
            if(act == ActivationFunction::RELU)
            {
                // Perform activation
                tmp = vmaxq_u8(vconst_0, vin);
                // Re-quantize to new output space
                tmp = vmlaq_qasymm8(tmp, vs, vo);
            }
            else if(act == ActivationFunction::BOUNDED_RELU)
            {
                // Perform activation
                tmp = vminq_u8(va, vmaxq_u8(vconst_0, vin));
                // Re-quantize to new output space
                tmp = vmlaq_qasymm8(tmp, vs, vo);
            }
            else if(act == ActivationFunction::LU_BOUNDED_RELU)
            {
                // Perform activation
                tmp = vminq_u8(va, vmaxq_u8(vb, vin));
                // Re-quantize to new output space
                tmp = vmlaq_qasymm8(tmp, vs, vo);
            }
            else if(act == ActivationFunction::LOGISTIC)
            {
                // De-quantize
                const auto vin_deq = vdequantize(vin, qi_in);
                // Perform activation
                const float32x4x4_t tmp_dep =
                {
                    {
                        wrapper::vdiv(vconst_1, wrapper::vadd(vconst_1, wrapper::vexpq(wrapper::vneg(vin_deq.val[0])))),
                        wrapper::vdiv(vconst_1, wrapper::vadd(vconst_1, wrapper::vexpq(wrapper::vneg(vin_deq.val[1])))),
                        wrapper::vdiv(vconst_1, wrapper::vadd(vconst_1, wrapper::vexpq(wrapper::vneg(vin_deq.val[2])))),
                        wrapper::vdiv(vconst_1, wrapper::vadd(vconst_1, wrapper::vexpq(wrapper::vneg(vin_deq.val[3])))),
                    }
                };
                // Re-quantize to new output space
                tmp = vquantize(tmp_dep, qi_out);
            }
            else
            {
                ARM_COMPUTE_ERROR("Unsupported activation function");
            }
            wrapper::vstore(output_ptr + x, tmp);
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            T in = *(reinterpret_cast<const T *>(input_ptr + x));
            T tmp;
            if(act == ActivationFunction::RELU)
            {
                tmp = std::max(const_0, in);
                tmp = std::max<int32_t>(0, std::min<int32_t>(tmp * s + o, 255));
            }
            else if(act == ActivationFunction::BOUNDED_RELU)
            {
                tmp = std::min(a, std::max(const_0, in));
                tmp = std::max<int32_t>(0, std::min<int32_t>(tmp * s + o, 255));
            }
            else if(act == ActivationFunction::LU_BOUNDED_RELU)
            {
                tmp = std::min(a, std::max(b, in));
                tmp = std::max<int32_t>(0, std::min<int32_t>(tmp * s + o, 255));
            }
            else if(act == ActivationFunction::LOGISTIC)
            {
                float tmp_f = scvt_f32_qasymm8(in, qi_in.scale, qi_in.offset);
                tmp_f       = 1.f / (1.f + std::exp(-tmp_f));
                tmp         = sqcvt_qasymm8_f32(tmp_f, qi_out.scale, qi_out.offset);
            }
            else
            {
                ARM_COMPUTE_ERROR("Unsupported activation function");
            }
            *(output_ptr + x) = tmp;
        }
    },
    input, output);
}

Status NEActivationLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(act_info);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), (output != nullptr) ? output->clone().get() : nullptr).first);

    return Status{};
}

void NEActivationLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window);
}
