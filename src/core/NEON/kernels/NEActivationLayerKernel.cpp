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
#include "arm_compute/core/NEON/kernels/NEActivationLayerKernel.h"

#include "arm_compute/core/FixedPoint.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEAsymm.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/NEON/NEMath.h"
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
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::QASYMM8, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);

    // Checks performed when output is configured
    if((output != nullptr) && (output->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    constexpr unsigned int num_elems_processed_per_iteration = 16;
    Window                 win                               = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
    bool                   window_changed                    = false;

    if(output != nullptr && (output->total_size() != 0))
    {
        AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);

        window_changed = update_window_and_padding(win,
                                                   AccessWindowHorizontal(input, 0, num_elems_processed_per_iteration),
                                                   output_access);

        output_access.set_valid_region(win, input->valid_region());
    }
    else
    {
        // In-place computation
        window_changed = update_window_and_padding(win,
                                                   AccessWindowHorizontal(input, 0, num_elems_processed_per_iteration));
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
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
        // Output auto inizialitation if not yet initialized
        auto_init_if_empty(*output->info(), *input->info()->clone());
        _output = output;
    }

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), (output != nullptr) ? output->info() : nullptr));

    ARM_COMPUTE_ERROR_ON_MSG((input->info()->data_type() == DataType::QASYMM8) && (activation_info.activation() != ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU),
                             "For QASYMM8 only lower/upper bounded relu is supported");

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
        { ActivationFunction::SOFT_RELU, &NEActivationLayerKernel::activation<ActivationFunction::SOFT_RELU, float16_t> },
        { ActivationFunction::SQRT, &NEActivationLayerKernel::activation<ActivationFunction::SQRT, float16_t> },
        { ActivationFunction::SQUARE, &NEActivationLayerKernel::activation<ActivationFunction::SQUARE, float16_t> },
        { ActivationFunction::TANH, &NEActivationLayerKernel::activation<ActivationFunction::TANH, float16_t> },
    };
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC*/

    // Activation functions : QS8
    static std::map<ActivationFunction, ActivationFunctionExecutorPtr> act_map_qs8 =
    {
        { ActivationFunction::ABS, &NEActivationLayerKernel::activation<ActivationFunction::ABS, qint8_t> },
        { ActivationFunction::LINEAR, &NEActivationLayerKernel::activation<ActivationFunction::LINEAR, qint8_t> },
        { ActivationFunction::LOGISTIC, &NEActivationLayerKernel::activation<ActivationFunction::LOGISTIC, qint8_t> },
        { ActivationFunction::RELU, &NEActivationLayerKernel::activation<ActivationFunction::RELU, qint8_t> },
        { ActivationFunction::BOUNDED_RELU, &NEActivationLayerKernel::activation<ActivationFunction::BOUNDED_RELU, qint8_t> },
        { ActivationFunction::LU_BOUNDED_RELU, &NEActivationLayerKernel::activation<ActivationFunction::LU_BOUNDED_RELU, qint8_t> },
        { ActivationFunction::LEAKY_RELU, &NEActivationLayerKernel::activation<ActivationFunction::LEAKY_RELU, qint8_t> },
        { ActivationFunction::SOFT_RELU, &NEActivationLayerKernel::activation<ActivationFunction::SOFT_RELU, qint8_t> },
        { ActivationFunction::SQRT, &NEActivationLayerKernel::activation<ActivationFunction::SQRT, qint8_t> },
        { ActivationFunction::SQUARE, &NEActivationLayerKernel::activation<ActivationFunction::SQUARE, qint8_t> },
        { ActivationFunction::TANH, &NEActivationLayerKernel::activation<ActivationFunction::TANH, qint8_t> },
    };
    // Activation functions : QS16
    static std::map<ActivationFunction, ActivationFunctionExecutorPtr> act_map_qs16 =
    {
        { ActivationFunction::ABS, &NEActivationLayerKernel::activation<ActivationFunction::ABS, qint16_t> },
        { ActivationFunction::LINEAR, &NEActivationLayerKernel::activation<ActivationFunction::LINEAR, qint16_t> },
        { ActivationFunction::LOGISTIC, &NEActivationLayerKernel::activation<ActivationFunction::LOGISTIC, qint16_t> },
        { ActivationFunction::RELU, &NEActivationLayerKernel::activation<ActivationFunction::RELU, qint16_t> },
        { ActivationFunction::BOUNDED_RELU, &NEActivationLayerKernel::activation<ActivationFunction::BOUNDED_RELU, qint16_t> },
        { ActivationFunction::LU_BOUNDED_RELU, &NEActivationLayerKernel::activation<ActivationFunction::LU_BOUNDED_RELU, qint16_t> },
        { ActivationFunction::LEAKY_RELU, &NEActivationLayerKernel::activation<ActivationFunction::LEAKY_RELU, qint16_t> },
        { ActivationFunction::SOFT_RELU, &NEActivationLayerKernel::activation<ActivationFunction::SOFT_RELU, qint16_t> },
        { ActivationFunction::SQRT, &NEActivationLayerKernel::activation<ActivationFunction::SQRT, qint16_t> },
        { ActivationFunction::SQUARE, &NEActivationLayerKernel::activation<ActivationFunction::SQUARE, qint16_t> },
        { ActivationFunction::TANH, &NEActivationLayerKernel::activation<ActivationFunction::TANH, qint16_t> },
    };
    // Activation functions : QASYMM8
    static std::map<ActivationFunction, ActivationFunctionExecutorPtr> act_map_qasymm8 =
    {
        { ActivationFunction::LU_BOUNDED_RELU, &NEActivationLayerKernel::activation<ActivationFunction::LU_BOUNDED_RELU, qasymm8_t> },
    };

    switch(input->info()->data_type())
    {
        case DataType::QASYMM8:
            _func = act_map_qasymm8[activation_info.activation()];
            break;
        case DataType::QS8:
            _func = act_map_qs8[activation_info.activation()];
            break;
        case DataType::QS16:
            _func = act_map_qs16[activation_info.activation()];
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

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <ActivationLayerInfo::ActivationFunction F, typename T>
typename std::enable_if<std::is_same<T, float16_t>::value, void>::type NEActivationLayerKernel::activation(const Window &window)
{
    Iterator input(_input, window);
    Iterator output(_output, window);

    static const float16x8_t CONST_0 = vdupq_n_f16(0.f);
    static const float16x8_t CONST_1 = vdupq_n_f16(1.f);

    const float16x8_t a = vdupq_n_f16(_act_info.a());
    const float16x8_t b = vdupq_n_f16(_act_info.b());

    execute_window_loop(window, [&](const Coordinates &)
    {
        const auto input_ptr  = reinterpret_cast<const float16_t *>(input.ptr());
        const auto output_ptr = reinterpret_cast<float16_t *>(output.ptr());

        const float16x8x2_t in  = vld2q_f16(input_ptr);
        float16x8x2_t       tmp = { {} };

        switch(F)
        {
            case ActivationFunction::ABS:
                tmp =
                {
                    {
                        vabsq_f16(in.val[0]),
                        vabsq_f16(in.val[1]),
                    }
                };
                break;
            case ActivationFunction::BOUNDED_RELU:
                tmp =
                {
                    {
                        vminq_f16(a, vmaxq_f16(CONST_0, in.val[0])),
                        vminq_f16(a, vmaxq_f16(CONST_0, in.val[1]))
                    }
                };
                break;
            case ActivationFunction::LU_BOUNDED_RELU:
                tmp =
                {
                    {
                        vminq_f16(a, vmaxq_f16(b, in.val[0])),
                        vminq_f16(a, vmaxq_f16(b, in.val[1]))
                    }
                };
                break;
            case ActivationFunction::LINEAR:
                tmp =
                {
                    {
                        vaddq_f16(b, vmulq_f16(a, in.val[0])),
                        vaddq_f16(b, vmulq_f16(a, in.val[1]))
                    }
                };
                break;
            case ActivationFunction::LOGISTIC:
                tmp =
                {
                    {
                        vinvq_f16(vaddq_f16(CONST_1, vexpq_f16(vnegq_f16(in.val[0])))),
                        vinvq_f16(vaddq_f16(CONST_1, vexpq_f16(vnegq_f16(in.val[1])))),
                    }
                };
                break;
            case ActivationFunction::RELU:
                tmp =
                {
                    {
                        vmaxq_f16(CONST_0, in.val[0]),
                        vmaxq_f16(CONST_0, in.val[1])
                    }
                };
                break;
            case ActivationFunction::LEAKY_RELU:
                tmp =
                {
                    {
                        vbslq_f16(vcgtq_f16(in.val[0], CONST_0), in.val[0], vmulq_f16(a, in.val[0])),
                        vbslq_f16(vcgtq_f16(in.val[1], CONST_0), in.val[1], vmulq_f16(a, in.val[1]))
                    }
                };
                break;
            case ActivationFunction::SOFT_RELU:
                tmp =
                {
                    {
                        vlogq_f16(vaddq_f16(CONST_1, vexpq_f16(in.val[0]))),
                        vlogq_f16(vaddq_f16(CONST_1, vexpq_f16(in.val[1]))),
                    }
                };
                break;
            case ActivationFunction::SQRT:
                tmp =
                {
                    {
                        vinvq_f16(vinvsqrtq_f16(in.val[0])),
                        vinvq_f16(vinvsqrtq_f16(in.val[1])),
                    }
                };
                break;
            case ActivationFunction::SQUARE:
                tmp =
                {
                    {
                        vmulq_f16(in.val[0], in.val[0]),
                        vmulq_f16(in.val[1], in.val[1])
                    }
                };
                break;
            case ActivationFunction::TANH:
                tmp =
                {
                    {
                        vmulq_f16(a, vtanhq_f16(vmulq_f16(b, in.val[0]))),
                        vmulq_f16(a, vtanhq_f16(vmulq_f16(b, in.val[1]))),
                    }
                };
                break;
            default:
                ARM_COMPUTE_ERROR("Not implemented");
                break;
        }

        vst2q_f16(output_ptr, tmp);
    },
    input, output);
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

template <ActivationLayerInfo::ActivationFunction F, typename T>
typename std::enable_if<std::is_same<T, float>::value, void>::type NEActivationLayerKernel::activation(const Window &window)
{
    Iterator input(_input, window);
    Iterator output(_output, window);

    static const float32x4_t CONST_1 = vdupq_n_f32(1.f);
    static const float32x4_t CONST_0 = vdupq_n_f32(0.f);
    const float32x4_t        a       = vdupq_n_f32(_act_info.a());
    const float32x4_t        b       = vdupq_n_f32(_act_info.b());

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto input_ptr  = reinterpret_cast<const float *>(input.ptr());
        const auto output_ptr = reinterpret_cast<float *>(output.ptr());

        const float32x4x4_t in  = vld4q_f32(input_ptr);
        float32x4x4_t       tmp = { {} };

        switch(F)
        {
            case ActivationFunction::ABS:
                tmp =
                {
                    {
                        vabsq_f32(in.val[0]),
                        vabsq_f32(in.val[1]),
                        vabsq_f32(in.val[2]),
                        vabsq_f32(in.val[3]),
                    }
                };
                break;
            case ActivationFunction::LINEAR:
                tmp =
                {
                    {
                        vmlaq_f32(b, a, in.val[0]),
                        vmlaq_f32(b, a, in.val[1]),
                        vmlaq_f32(b, a, in.val[2]),
                        vmlaq_f32(b, a, in.val[3]),
                    }
                };
                break;
            case ActivationFunction::LOGISTIC:
                tmp =
                {
                    {
                        vinvq_f32(vaddq_f32(CONST_1, vexpq_f32(vnegq_f32(in.val[0])))),
                        vinvq_f32(vaddq_f32(CONST_1, vexpq_f32(vnegq_f32(in.val[1])))),
                        vinvq_f32(vaddq_f32(CONST_1, vexpq_f32(vnegq_f32(in.val[2])))),
                        vinvq_f32(vaddq_f32(CONST_1, vexpq_f32(vnegq_f32(in.val[3])))),
                    }
                };
                break;
            case ActivationFunction::RELU:
                tmp =
                {
                    {
                        vmaxq_f32(CONST_0, in.val[0]),
                        vmaxq_f32(CONST_0, in.val[1]),
                        vmaxq_f32(CONST_0, in.val[2]),
                        vmaxq_f32(CONST_0, in.val[3]),
                    }
                };
                break;
            case ActivationFunction::BOUNDED_RELU:
                tmp =
                {
                    {
                        vminq_f32(a, vmaxq_f32(CONST_0, in.val[0])),
                        vminq_f32(a, vmaxq_f32(CONST_0, in.val[1])),
                        vminq_f32(a, vmaxq_f32(CONST_0, in.val[2])),
                        vminq_f32(a, vmaxq_f32(CONST_0, in.val[3])),
                    }
                };
                break;
            case ActivationFunction::LU_BOUNDED_RELU:
                tmp =
                {
                    {
                        vminq_f32(a, vmaxq_f32(b, in.val[0])),
                        vminq_f32(a, vmaxq_f32(b, in.val[1])),
                        vminq_f32(a, vmaxq_f32(b, in.val[2])),
                        vminq_f32(a, vmaxq_f32(b, in.val[3])),
                    }
                };
                break;
            case ActivationFunction::LEAKY_RELU:
                tmp =
                {
                    {
                        vbslq_f32(vcgtq_f32(in.val[0], CONST_0), in.val[0], vmulq_f32(a, in.val[0])),
                        vbslq_f32(vcgtq_f32(in.val[1], CONST_0), in.val[1], vmulq_f32(a, in.val[1])),
                        vbslq_f32(vcgtq_f32(in.val[2], CONST_0), in.val[2], vmulq_f32(a, in.val[2])),
                        vbslq_f32(vcgtq_f32(in.val[3], CONST_0), in.val[3], vmulq_f32(a, in.val[3])),
                    }
                };
                break;
            case ActivationFunction::SOFT_RELU:
                tmp =
                {
                    {
                        vlogq_f32(vaddq_f32(CONST_1, vexpq_f32(in.val[0]))),
                        vlogq_f32(vaddq_f32(CONST_1, vexpq_f32(in.val[1]))),
                        vlogq_f32(vaddq_f32(CONST_1, vexpq_f32(in.val[2]))),
                        vlogq_f32(vaddq_f32(CONST_1, vexpq_f32(in.val[3]))),
                    }
                };
                break;
            case ActivationFunction::SQRT:
                tmp =
                {
                    {
                        vinvq_f32(vinvsqrtq_f32(in.val[0])),
                        vinvq_f32(vinvsqrtq_f32(in.val[1])),
                        vinvq_f32(vinvsqrtq_f32(in.val[2])),
                        vinvq_f32(vinvsqrtq_f32(in.val[3])),
                    }
                };
                break;
            case ActivationFunction::SQUARE:
                tmp =
                {
                    {
                        vmulq_f32(in.val[0], in.val[0]),
                        vmulq_f32(in.val[1], in.val[1]),
                        vmulq_f32(in.val[2], in.val[2]),
                        vmulq_f32(in.val[3], in.val[3]),
                    }
                };
                break;
            case ActivationFunction::TANH:
                tmp =
                {
                    {
                        vmulq_f32(a, vtanhq_f32(vmulq_f32(b, in.val[0]))),
                        vmulq_f32(a, vtanhq_f32(vmulq_f32(b, in.val[1]))),
                        vmulq_f32(a, vtanhq_f32(vmulq_f32(b, in.val[2]))),
                        vmulq_f32(a, vtanhq_f32(vmulq_f32(b, in.val[3]))),
                    }
                };
                break;
            default:
                break;
        }

        vst4q_f32(output_ptr, tmp);
    },
    input, output);
}

template <ActivationLayerInfo::ActivationFunction F, typename T>
typename std::enable_if<std::is_same<T, int8_t>::value, void>::type NEActivationLayerKernel::activation(const Window &window)
{
    Iterator  input(_input, window);
    Iterator  output(_output, window);
    const int fixed_point_position = _input->info()->fixed_point_position();

    static const qint8x16_t CONST_0 = vdupq_n_qs8(0);
    const qint8x16_t        CONST_1 = vdupq_n_qs8(sqcvt_qs8_f32(1.f, fixed_point_position));
    const qint8x16_t        a       = vdupq_n_qs8(sqcvt_qs8_f32(_act_info.a(), fixed_point_position));
    const qint8x16_t        b       = vdupq_n_qs8(sqcvt_qs8_f32(_act_info.b(), fixed_point_position));

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto input_ptr  = reinterpret_cast<const int8_t *>(input.ptr());
        const auto output_ptr = reinterpret_cast<int8_t *>(output.ptr());

        const qint8x16_t in  = vld1q_qs8(input_ptr);
        qint8x16_t       tmp = {};

        switch(F)
        {
            case ActivationFunction::ABS:
                tmp = vqabsq_qs8(in);
                break;
            case ActivationFunction::LINEAR:
                tmp = vqmlaq_qs8(b, a, in, fixed_point_position);
                break;
            case ActivationFunction::LOGISTIC:
                tmp = vqrecipq_qs8(vqaddq_qs8(CONST_1, vqexpq_qs8(vnegq_s8(in), fixed_point_position)), fixed_point_position);
                break;
            case ActivationFunction::RELU:
                tmp = vmaxq_qs8(CONST_0, in);
                break;
            case ActivationFunction::BOUNDED_RELU:
                tmp = vminq_qs8(a, vmaxq_qs8(CONST_0, in));
                break;
            case ActivationFunction::LU_BOUNDED_RELU:
                tmp = vminq_qs8(a, vmaxq_qs8(b, in));
                break;
            case ActivationFunction::LEAKY_RELU:
                tmp = vbslq_s8(vcgtq_s8(in, CONST_0), in, vmulq_qs8(a, in, fixed_point_position));
                break;
            case ActivationFunction::SOFT_RELU:
                tmp = vlogq_qs8(vqaddq_qs8(CONST_1, vqexpq_qs8(in, fixed_point_position)), fixed_point_position);
                break;
            case ActivationFunction::SQRT:
                tmp = vqrecipq_qs8(vqinvsqrtq_qs8(in, fixed_point_position), fixed_point_position);
                break;
            case ActivationFunction::SQUARE:
                tmp = vqmulq_qs8(in, in, fixed_point_position);
                break;
            case ActivationFunction::TANH:
                tmp = vqmulq_qs8(a, vqtanhq_qs8(vqmulq_qs8(b, in, fixed_point_position), fixed_point_position), fixed_point_position);
                break;
            default:
                break;
        }

        vst1q_qs8(output_ptr, tmp);
    },
    input, output);
}

template <ActivationLayerInfo::ActivationFunction F, typename T>
typename std::enable_if<std::is_same<T, qasymm8_t>::value, void>::type NEActivationLayerKernel::activation(const Window &window)
{
    Iterator               input(_input, window);
    Iterator               output(_output, window);
    const QuantizationInfo qi_in  = _input->info()->quantization_info();
    const QuantizationInfo qi_out = _output->info()->quantization_info();
    const qasymm8x16_t     a      = vdupq_n_u8(sqcvt_qasymm8_f32(_act_info.a(), qi_in.scale, qi_in.offset));
    const qasymm8x16_t     b      = vdupq_n_u8(sqcvt_qasymm8_f32(_act_info.b(), qi_in.scale, qi_in.offset));
    // Initialise scale/offset for re-quantization
    float       s  = qi_in.scale / qi_out.scale;
    float       o  = -qi_in.offset * s + qi_out.offset;
    float32x4_t vs = vdupq_n_f32(s);
    float32x4_t vo = vdupq_n_f32(o);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto input_ptr  = reinterpret_cast<const qasymm8_t *>(input.ptr());
        const auto output_ptr = reinterpret_cast<qasymm8_t *>(output.ptr());

        const qasymm8x16_t in  = vld1q_u8(input_ptr);
        qasymm8x16_t       tmp = {};

        switch(F)
        {
            case ActivationFunction::LU_BOUNDED_RELU:
                // Perform activation
                tmp = vminq_u8(a, vmaxq_u8(b, in));
                // Re-quantize to new output space
                tmp = vmlaq_qasymm8(tmp, vs, vo);
                break;
            default:
                ARM_COMPUTE_ERROR("Function not implemented");
                break;
        }

        vst1q_u8(output_ptr, tmp);
    },
    input, output);
}

template <ActivationLayerInfo::ActivationFunction F, typename T>
typename std::enable_if<std::is_same<T, qint16_t>::value, void>::type NEActivationLayerKernel::activation(const Window &window)
{
    Iterator  input(_input, window);
    Iterator  output(_output, window);
    const int fixed_point_position = _input->info()->fixed_point_position();

    static const qint16x8_t CONST_0 = vdupq_n_qs16(0);
    const qint16x8_t        CONST_1 = vdupq_n_qs16(sqcvt_qs16_f32(1.f, fixed_point_position));
    const qint16x8_t        a       = vdupq_n_qs16(sqcvt_qs16_f32(_act_info.a(), fixed_point_position));
    const qint16x8_t        b       = vdupq_n_qs16(sqcvt_qs16_f32(_act_info.b(), fixed_point_position));

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto input_ptr  = reinterpret_cast<const int16_t *>(input.ptr());
        const auto output_ptr = reinterpret_cast<int16_t *>(output.ptr());

        const qint16x8x2_t in  = vld2q_s16(input_ptr);
        qint16x8x2_t       tmp = { {} };

        switch(F)
        {
            case ActivationFunction::ABS:
                tmp =
                {
                    {
                        vqabsq_qs16(in.val[0]),
                        vqabsq_qs16(in.val[1]),
                    }
                };
                break;
            case ActivationFunction::LINEAR:
                tmp =
                {
                    {
                        vqmlaq_qs16(b, a, in.val[0], fixed_point_position),
                        vqmlaq_qs16(b, a, in.val[1], fixed_point_position),
                    }
                };
                break;
            case ActivationFunction::LOGISTIC:
                tmp =
                {
                    {
                        vqrecipq_qs16(vqaddq_qs16(CONST_1, vqexpq_qs16(vnegq_s16(in.val[0]), fixed_point_position)), fixed_point_position),
                        vqrecipq_qs16(vqaddq_qs16(CONST_1, vqexpq_qs16(vnegq_s16(in.val[1]), fixed_point_position)), fixed_point_position),
                    }
                };
                break;
            case ActivationFunction::RELU:
                tmp =
                {
                    {
                        vmaxq_qs16(CONST_0, in.val[0]),
                        vmaxq_qs16(CONST_0, in.val[1]),
                    }
                };
                break;
            case ActivationFunction::BOUNDED_RELU:
                tmp =
                {
                    {
                        vminq_qs16(a, vmaxq_qs16(CONST_0, in.val[0])),
                        vminq_qs16(a, vmaxq_qs16(CONST_0, in.val[1])),
                    }
                };
                break;
            case ActivationFunction::LU_BOUNDED_RELU:
                tmp =
                {
                    {
                        vminq_qs16(a, vmaxq_qs16(b, in.val[0])),
                        vminq_qs16(a, vmaxq_qs16(b, in.val[1])),
                    }
                };
                break;
            case ActivationFunction::LEAKY_RELU:
                tmp =
                {
                    {
                        vbslq_s16(vcgtq_s16(in.val[0], CONST_0), in.val[0], vmulq_qs16(a, in.val[0], fixed_point_position)),
                        vbslq_s16(vcgtq_s16(in.val[1], CONST_0), in.val[1], vmulq_qs16(a, in.val[1], fixed_point_position)),
                    }
                };
                break;
            case ActivationFunction::SOFT_RELU:
                tmp =
                {
                    {
                        vlogq_qs16(vqaddq_qs16(CONST_1, vqexpq_qs16(in.val[0], fixed_point_position)), fixed_point_position),
                        vlogq_qs16(vqaddq_qs16(CONST_1, vqexpq_qs16(in.val[1], fixed_point_position)), fixed_point_position),
                    }
                };
                break;
            case ActivationFunction::SQRT:
                tmp =
                {
                    {
                        vqrecipq_qs16(vqinvsqrtq_qs16(in.val[0], fixed_point_position), fixed_point_position),
                        vqrecipq_qs16(vqinvsqrtq_qs16(in.val[1], fixed_point_position), fixed_point_position),
                    }
                };
                break;
            case ActivationFunction::SQUARE:
                tmp =
                {
                    {
                        vqmulq_qs16(in.val[0], in.val[0], fixed_point_position),
                        vqmulq_qs16(in.val[1], in.val[1], fixed_point_position),
                    }
                };
                break;
            case ActivationFunction::TANH:
                tmp =
                {
                    {
                        vqmulq_qs16(a, vqtanhq_qs16(vqmulq_qs16(b, in.val[0], fixed_point_position), fixed_point_position), fixed_point_position),
                        vqmulq_qs16(a, vqtanhq_qs16(vqmulq_qs16(b, in.val[1], fixed_point_position), fixed_point_position), fixed_point_position),
                    }
                };
                break;
            default:
                ARM_COMPUTE_ERROR("Function not implemented");
                break;
        }

        vst2q_qs16(output_ptr, tmp);
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
