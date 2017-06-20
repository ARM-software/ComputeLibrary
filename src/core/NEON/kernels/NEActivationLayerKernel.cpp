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
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/NEON/NEMath.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>
#include <array>
#include <cmath>
#include <map>

using namespace arm_compute;

NEActivationLayerKernel::NEActivationLayerKernel()
    : _input(nullptr), _output(nullptr), _func(nullptr), _act_info(ActivationFunction::LOGISTIC)
{
}

void NEActivationLayerKernel::configure(ITensor *input, ITensor *output, ActivationLayerInfo activation_info)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32, DataType::QS8);

    _input    = input;
    _act_info = activation_info;
    _output   = input;

    if(output != nullptr)
    {
        // Output auto inizialitation if not yet initialized
        auto_init_if_empty(*output->info(), input->info()->tensor_shape(), 1, input->info()->data_type(), input->info()->fixed_point_position());

        ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);

        _output = output;
    }

    // Activation functions : FP32
    static std::map<ActivationFunction, ActivationFunctionExecutorPtr> act_map_f32 =
    {
        { ActivationFunction::ABS, &NEActivationLayerKernel::activation<ActivationFunction::ABS, float> },
        { ActivationFunction::LINEAR, &NEActivationLayerKernel::activation<ActivationFunction::LINEAR, float> },
        { ActivationFunction::LOGISTIC, &NEActivationLayerKernel::activation<ActivationFunction::LOGISTIC, float> },
        { ActivationFunction::RELU, &NEActivationLayerKernel::activation<ActivationFunction::RELU, float> },
        { ActivationFunction::BOUNDED_RELU, &NEActivationLayerKernel::activation<ActivationFunction::BOUNDED_RELU, float> },
        { ActivationFunction::SOFT_RELU, &NEActivationLayerKernel::activation<ActivationFunction::SOFT_RELU, float> },
        { ActivationFunction::SQRT, &NEActivationLayerKernel::activation<ActivationFunction::SQRT, float> },
        { ActivationFunction::SQUARE, &NEActivationLayerKernel::activation<ActivationFunction::SQUARE, float> },
        { ActivationFunction::TANH, &NEActivationLayerKernel::activation<ActivationFunction::TANH, float> },
    };

    // Activation functions : QS8
    static std::map<ActivationFunction, ActivationFunctionExecutorPtr> act_map_qs8 =
    {
        { ActivationFunction::ABS, &NEActivationLayerKernel::activation<ActivationFunction::ABS, qint8_t> },
        { ActivationFunction::LINEAR, &NEActivationLayerKernel::activation<ActivationFunction::LINEAR, qint8_t> },
        { ActivationFunction::LOGISTIC, &NEActivationLayerKernel::activation<ActivationFunction::LOGISTIC, qint8_t> },
        { ActivationFunction::RELU, &NEActivationLayerKernel::activation<ActivationFunction::RELU, qint8_t> },
        { ActivationFunction::BOUNDED_RELU, &NEActivationLayerKernel::activation<ActivationFunction::BOUNDED_RELU, qint8_t> },
        { ActivationFunction::SOFT_RELU, &NEActivationLayerKernel::activation<ActivationFunction::SOFT_RELU, qint8_t> },
        { ActivationFunction::SQRT, &NEActivationLayerKernel::activation<ActivationFunction::SQRT, qint8_t> },
        { ActivationFunction::SQUARE, &NEActivationLayerKernel::activation<ActivationFunction::SQUARE, qint8_t> },
        { ActivationFunction::TANH, &NEActivationLayerKernel::activation<ActivationFunction::TANH, qint8_t> },
    };

    switch(input->info()->data_type())
    {
        case DataType::F32:
            _func = act_map_f32[activation_info.activation()];
            break;
        case DataType::QS8:
            _func = act_map_qs8[activation_info.activation()];
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported data type.");
    }

    constexpr unsigned int num_elems_processed_per_iteration = 16;

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    if(output != nullptr)
    {
        AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

        update_window_and_padding(win,
                                  AccessWindowHorizontal(input->info(), 0, num_elems_processed_per_iteration),
                                  output_access);

        output_access.set_valid_region(win, input->info()->valid_region());
    }
    else
    {
        // In-place computation
        update_window_and_padding(win,
                                  AccessWindowHorizontal(input->info(), 0, num_elems_processed_per_iteration));
    }

    ICPPKernel::configure(win);
}

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
    Iterator input(_input, window);
    Iterator output(_output, window);
    int      fixed_point_position = _input->info()->fixed_point_position();

    static const qint8x16_t CONST_0 = vdupq_n_qs8(0);
    const qint8x16_t        CONST_1 = vdupq_n_qs8(scvt_qs8_f32(1.f, fixed_point_position));
    const qint8x16_t        a       = vdupq_n_qs8(scvt_qs8_f32(_act_info.a(), fixed_point_position));
    const qint8x16_t        b       = vdupq_n_qs8(scvt_qs8_f32(_act_info.b(), fixed_point_position));

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
            case ActivationFunction::BOUNDED_RELU:
                tmp = vminq_qs8(a, vmaxq_qs8(CONST_0, in));
                break;
            case ActivationFunction::LINEAR:
                tmp = vqmlaq_qs8(b, a, in, fixed_point_position);
                break;
            case ActivationFunction::LOGISTIC:
                tmp = vrecipq_qs8(vqaddq_qs8(CONST_1, vqexpq_qs8(vnegq_s8(in), fixed_point_position)), fixed_point_position);
                break;
            case ActivationFunction::RELU:
                tmp = vmaxq_qs8(CONST_0, in);
                break;
            case ActivationFunction::SOFT_RELU:
                tmp = vlogq_qs8(vqaddq_qs8(CONST_1, vqexpq_qs8(in, fixed_point_position)), fixed_point_position);
                break;
            case ActivationFunction::SQRT:
                tmp = vrecipq_qs8(vinvsqrtq_qs8(in, fixed_point_position), fixed_point_position);
                break;
            case ActivationFunction::SQUARE:
                tmp = vqmulq_qs8(in, in, fixed_point_position);
                break;
            case ActivationFunction::TANH:
                tmp = vtanhq_qs8(in, fixed_point_position);
                break;
            default:
                break;
        }

        vst1q_qs8(output_ptr, tmp);
    },
    input, output);
}

void NEActivationLayerKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window);
}
