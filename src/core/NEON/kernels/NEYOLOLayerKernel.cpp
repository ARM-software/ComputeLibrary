/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "arm_compute/core/NEON/kernels/NEYOLOLayerKernel.h"

#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "src/core/NEON/NEAsymm.h"
#include "src/core/NEON/NEFixedPoint.h"
#include "src/core/NEON/NEMath.h"

#include "src/core/NEON/kernels/detail/NEActivationFunctionDetail.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const ActivationLayerInfo &act_info, int32_t num_classes)
{
    ARM_COMPUTE_UNUSED(act_info);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_layout() == DataLayout::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.activation() != ActivationLayerInfo::ActivationFunction::LOGISTIC);

    const unsigned int channel_idx = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::CHANNEL);
    ARM_COMPUTE_RETURN_ERROR_ON(num_classes <= 0);
    ARM_COMPUTE_RETURN_ERROR_ON((input->dimension(channel_idx) % (num_classes + 5)) != 0);

    // Checks performed when output is configured
    if((output != nullptr) && (output->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}
} // namespace

NEYOLOLayerKernel::NEYOLOLayerKernel()
    : _func(nullptr), _input(nullptr), _output(nullptr), _act_info(), _num_classes()
{
}

template <typename T, int S>
void NEYOLOLayerKernel::yolo_layer_nchw(const Window &window)
{
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());
    const int  window_step_x  = S;

    Window win{ window };
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    Iterator input(_input, win);
    Iterator output(_output, win);

    execute_window_loop(win, [&](const Coordinates & id)
    {
        const auto input_ptr  = reinterpret_cast<const T *>(input.ptr());
        const auto output_ptr = reinterpret_cast<T *>(output.ptr());
        int        x          = window_start_x;
        const int  box_ch_id  = id.z() % (_num_classes + 5);
        const bool activate   = box_ch_id != 2 && box_ch_id != 3;

        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            auto res = wrapper::vloadq(input_ptr + x);

            // Perform activation
            if(activate)
            {
                auto activation = detail::logistic<T, S>(_act_info);
                activation(res);
            }

            // Store results
            wrapper::vstore(output_ptr + x, res);
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            auto res = *(input_ptr + x);

            // Perform activation
            if(activate)
            {
                res = 1.f / (1.f + std::exp(-res));
            }

            *(output_ptr + x) = res;
        }
    },
    input, output);
}

template <typename T>
void NEYOLOLayerKernel::yolo_layer_nhwc(const Window &window)
{
    Iterator input(_input, window);
    Iterator output(_output, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        auto res = *(reinterpret_cast<T *>(input.ptr()));

        const int  box_ch_id = id.x() % (_num_classes + 5);
        const bool activate  = box_ch_id != 2 && box_ch_id != 3;

        // Perform activation
        if(activate)
        {
            res = 1.f / (1.f + std::exp(-res));
        }

        // Store result
        *(reinterpret_cast<T *>(output.ptr())) = res;
    },
    input, output);
}

void NEYOLOLayerKernel::configure(ITensor *input, ITensor *output, const ActivationLayerInfo &act_info, int32_t num_classes)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), (output != nullptr) ? output->info() : nullptr, act_info, num_classes));

    _input       = input;
    _output      = output;
    _act_info    = act_info;
    _num_classes = num_classes;

    switch(_input->info()->data_type())
    {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            _func = (_input->info()->data_layout() == DataLayout::NHWC) ? &NEYOLOLayerKernel::yolo_layer_nhwc<float16_t> : &NEYOLOLayerKernel::yolo_layer_nchw<float16_t, 8>;
            break;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F32:
            _func = (_input->info()->data_layout() == DataLayout::NHWC) ? &NEYOLOLayerKernel::yolo_layer_nhwc<float> : &NEYOLOLayerKernel::yolo_layer_nchw<float, 4>;
            break;
        default:
            ARM_COMPUTE_ERROR("Element size not supported");
            break;
    }

    Window win = calculate_max_window(*input->info(), Steps());

    // Configure kernel window
    if(output != nullptr)
    {
        // Output auto inizialitation if not yet initialized
        auto_init_if_empty(*output->info(), *input->info());

        Coordinates coord;
        coord.set_num_dimensions(output->info()->num_dimensions());

        output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));
    }

    ICPPKernel::configure(win);
}

Status NEYOLOLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const ActivationLayerInfo &act_info, int32_t num_classes)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, act_info, num_classes));

    return Status{};
}

void NEYOLOLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window);
}
} // namespace arm_compute
